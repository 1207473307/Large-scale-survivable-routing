import networkx as nx
from collections import deque
from Date_processing import *
import matplotlib.pyplot as plt
import time
import os
from networkx.readwrite import json_graph
import json
import pickle

path='./subgraphs/'


# 检查链路上的中继节点是否满足边际值要求的函数
def check_margin(G, path):
    total_distance = 0
    ots = 0
    total_osnr = 0

    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        total_distance += min([e['distance'] for e in G[current_node][next_node].values()])
        ots += min([e['ots'] for e in G[current_node][next_node].values()])
        total_osnr += min([e['osnr'] for e in G[current_node][next_node].values()])
        # # 假设G是多重图，我们需要找到正确的边
        # for key, attr in self.G[current_node][next_node].items():
        #     total_distance += attr['distance']
        #     ots += attr['ots']
        #     min_osnr += attr['osnr']

    # 检查总距离、OTS跳数和最小OSNR是否在边际值内
    return (total_distance <= distance_margin and
            ots <= ots_margin and
            total_osnr <= osnr_margin
            )

def can_join(G, domain, edge):
    k = 1
    u, v = edge
    domain_nodes = domain['nodes']

    # 检查 u 节点到 domain 中每个节点的路径
    for domain_node in domain_nodes:
        # print(domain_node)
        if not nx.has_path(G, u, domain_node):
            return False
        path = nx.dijkstra_path(G, u, domain_node, weight='distance')
        if not check_margin(G, path):
            return False
        # paths = nx.all_shortest_paths(G, u, domain_node, weight='distance')
        # if not any(check_margin(G, path) for path in list(paths)[:k]):
        #     return False

    # 检查 v 节点到 domain 中每个节点的路径
    for domain_node in domain_nodes:
        if not nx.has_path(G, v, domain_node):
            return False
        path = nx.dijkstra_path(G, v, domain_node, weight='distance')
        if not check_margin(G, path):
            return False
        # paths = nx.all_shortest_paths(G, v, domain_node, weight='distance')
        # if not any(check_margin(G, path) for path in list(paths)[:k]):
        #     return False

    return True


def process_unassigned_nodes_edges(G, domains):
    unassigned_edges = set(G.edges()) - {edge for domain in domains for edge in domain['edges']}
    # unassigned_nodes = set(G.nodes()) - {node for domain in domains for node in domain['nodes']}

    while unassigned_edges:
        print('number of unassigned edges:', len(unassigned_edges))
        edge = unassigned_edges.pop()
        assigned = False
        for domain in domains:
            if can_join(G, domain, edge):
                domain['edges'].add(edge)
                domain['nodes'].update(edge)
                if G.nodes[edge[0]]['relay']:
                    domain['relay_nodes'].add(edge[0])
                if G.nodes[edge[1]]['relay']:
                    domain['relay_nodes'].add(edge[1])
                assigned = True
                break

        if not assigned:
            new_domain = {'edges': {edge}, 'nodes': set(edge), 'relay_nodes': set()}
            if G.nodes[edge[0]]['relay']:
                new_domain['relay_nodes'].add(edge[0])
            if G.nodes[edge[1]]['relay']:
                new_domain['relay_nodes'].add(edge[1])
            domains.append(new_domain)
    return domains


def merge_domains(domains):
    merged = True
    while merged:
        merged = False
        new_domains = []
        while domains:
            domain = domains.pop(0)
            merged_domain = domain
            merge_indices = []

            for i, other_domain in enumerate(domains):
            #     if merged_domain['relay_nodes'] & other_domain['relay_nodes']:
                if merged_domain['relay_nodes'].issubset(other_domain['relay_nodes']) \
                        or merged_domain['relay_nodes'].issuperset(other_domain['relay_nodes']):
                    # 合并两个 domain
                    merged_domain['edges'].update(other_domain['edges'])
                    merged_domain['nodes'].update(other_domain['nodes'])
                    merged_domain['relay_nodes'].update(other_domain['relay_nodes'])
                    merge_indices.append(i)
                    merged = True

            # 从 domains 中移除已经合并的 domain
            for index in sorted(merge_indices, reverse=True):
                domains.pop(index)

            new_domains.append(merged_domain)
        domains = new_domains
    return domains

def create_subgraphs(G, domains):
    subgraphs = []
    for domain in domains:
        subgraph = nx.MultiGraph()
        # 添加节点
        subgraph.add_nodes_from(domain['nodes'])
        # 添加边
        for edge in domain['edges']:
            u, v = edge
            for key, attr in G[u][v].items():
                subgraph.add_edge(u, v, key=key, **attr)
        subgraphs.append(subgraph)
    return subgraphs

def greedy_cluster(G, file_path, file_name):
    # subgraphs = []
    domains = []
    relay_nodes = set()
    all_nodes = set()
    service_success_route_recorder = read_pkl_file(file_path + file_name + 'service_success_route_recorder' + '.pkl')
    for service_route in service_success_route_recorder:
        for sub_route in service_route:

            route = sub_route['route']
            route_nodes = set(route)
            route_edges = set([(route[i], route[i+1]) for i in range(len(route)-1)])
            route_relay_node = -1

            all_nodes.update(route_nodes)
            if sub_route['relay_index'] != -1:
                relay_nodes.update([sub_route['route'][-1]])
                route_relay_node = sub_route['route'][-1]
            flag = 1
            for domain in domains:
                if route_edges & domain['edges']:
                    domain['nodes'].update(route_nodes)
                    domain['edges'].update(route_edges)
                    if route_relay_node != -1:
                        domain['relay_nodes'].update([route_relay_node])

                    flag = 0
                    break
            if flag:
                domain = {'edges': route_edges, 'nodes': route_nodes}
                if route_relay_node != -1:
                    domain['relay_nodes'] = {route_relay_node}
                else:
                    domain['relay_nodes'] = set()
                domains.append(domain)

    # # 合并
    # merged_domains = merge_domains(domains)
    #
    # # 处理为分配的边和节点
    # domains = process_unassigned_nodes_edges(G, merged_domains)
    domains = process_unassigned_nodes_edges(G, domains)
    domains = merge_domains(domains)
    print('number of domains:', len(domains))

    subgraphs = create_subgraphs(G, domains)



    # 记录每个子图中可作为中继节点的具体节点
    for i in range(len(subgraphs)):
        subgraphs[i].graph['relay_in_subgraphs'] = []
        subgraphs[i].graph['subgraph_id'] = i
        for n in subgraphs[i].nodes():
            if G.nodes[n]['relay'] == True:
                subgraphs[i].graph['relay_in_subgraphs'].append([n, G.nodes[n]['available relay num']])
    # 保存子图
    if not os.path.exists('subgraphs/' + file_name + 'greedy'):
        os.makedirs('subgraphs/' + file_name + 'greedy')
    for i in range(len(subgraphs)):
        save_pkl('subgraphs/' + file_name + 'greedy' + '/' + 'sg' + str(i) + '.pkl', subgraphs[i])
    return subgraphs


def save_pkl(filename, G):
    with open(filename, 'wb') as f:
        pickle.dump(G, f)


def read_pkl_file(filename):
    with open(filename, 'rb') as f:
        G = pickle.load(f)
    return G


def load_sg(path):
    subgraphs = []
    num_files = 0
    for root, dirs, files in os.walk(path):
        num_files += len(files)
    for i in range(num_files):
        sg = read_pkl_file(path+'sg'+str(i)+'.pkl')
        subgraphs.append(sg)
    return subgraphs


if __name__ == '__main__':
    # file_name='example1'
    # G=create_topology(file_path='example/',file_name=file_name,band=24)

    file_name = 'example2'
    G = create_topology(file_path='example/', file_name=file_name, band=8)


    distance_margin=800
    ots_margin=10
    osnr_margin=0.01
    # subgraphs = []
    # start_time = time.time()
    # subgraphs = cluster(G,distance_margin,ots_margin,osnr_margin,file_name)
    # end_time = time.time()
    # subgraphs=load_sg(path+file_name+'/')
    # print("Time:", end_time - start_time)
    # print("end")
    subgraphs = greedy_cluster(G, file_path='example/',file_name=file_name)
    print(subgraphs)