import networkx as nx
import random
import time
import numpy as np
from Date_processing import *
from cluster import *
from route_layering import RouteLayering
import matplotlib.pyplot as plt
from queue import PriorityQueue
import itertools
import pickle

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


seed = 1
set_seed(seed)


class ServiceRecovery():
    def __init__(self, distance_margin, ots_margin, osnr_margin, subgraph_file_path, file_path='example/', file_name='example1', band=24):
        self.distance_margin = distance_margin
        self.ots_margin = ots_margin
        self.osnr_margin = osnr_margin

        self.G = create_topology(file_path, file_name, band)  # 拓扑
        # self.subgraphs = cluster(self.G, self.distance_margin, self.ots_margin, self.osnr_margin, file_name)
        self.subgraphs = load_sg(subgraph_file_path)
        # self.random_add_layer_edge(num=int(self.N/2))
        self.services = process_service(file_path, file_name, band)

        self.N = self.G.number_of_nodes()  # 节点数量
        self.M = self.G.number_of_edges()  # 边数量
        # self.L = len(self.G.edges()[0]['f_slot']) # 频隙（波）数量
        self.L = self.G.graph['L']
        # self.G_layering = self.creat_g_layering()  # 分层辅助图
        self.subgraphs_layering = self.creat_sub_layering()  # [k][l] 第k个子图第l层分层辅助图
        # self.util_rate = util_rate  # 网络利用率（占用率）

        # 处理节点在哪个域
        self.process_domain()

    @staticmethod
    # 路由计算
    def route(G, s, d, k=1):
        if k == 1:
            if nx.has_path(G, s, d):
                return nx.dijkstra_path(G, s, d, weight='cost')
            else:
                return 0

    @staticmethod
    def resource_occupation(G):
        free_slot = 0
        num_e = 0
        for u, v in G.edges():
            for e in G[u][v].values():
                num_e += 1
                free_slot += sum(e['f_slot'])
                L_slot = len(e['f_slot'])
        return 1 - free_slot / (num_e * L_slot)

    # 生成分层辅助图 for subgraph
    def creat_g_layering(self, subgraph):
        # network = nx.DiGraph()

        subgraph_layering_list = []
        for l in range(self.L):
            network = nx.MultiGraph()
            network.add_nodes_from(subgraph.nodes())
            for u, v in subgraph.edges():
                edge_list = []
                for e in self.G[u][v].values():
                    if e['f_slot'][l] == 1:
                        edge_list.append((u, v, e))
                network.add_edges_from(edge_list)

            subgraph_layering_list.append(network)

        return subgraph_layering_list

    def creat_sub_layering(self):
        # print(self.G.edges())
        # a = self.G[15][332]
        # print(self.G[15][332])
        subgraphs_layering = []  # [k][l] 第k个子图第l层分层辅助图
        for subgraph in self.subgraphs:
            subgraphs_layering.append(self.creat_g_layering(subgraph))
        return subgraphs_layering

    # 处理节点在哪个域
    def process_domain(self):
        for n in self.G.nodes():
            self.G.nodes[n]['domain'] = set()
            for i, sg in enumerate(self.subgraphs):
                if sg.has_node(n):
                    self.G.nodes[n]['domain'].update([i])


    # 更新分层辅助图（部署请求后）
    def update_g_layering(self):
        pass

    def construct_G_sub(self, G, subgraphs):
        G_sub = nx.MultiGraph()
        G_sub_graph = nx.Graph()
        for i in range(len(subgraphs)):
            sg = subgraphs[i]
            G_sub.add_node(i, relay_node=sg.graph['relay_in_subgraphs'])
            G_sub_graph.add_node(i, relay_node=sg.graph['relay_in_subgraphs'])
        # 多个节点的子图处理
        for i, j in itertools.combinations(G_sub.nodes(), 2):
            for m in G_sub.nodes[i]['relay_node']:
                for n in G_sub.nodes[j]['relay_node']:
                    # if G.has_edge(m[0], n[0]) and m[1] * n[1]:
                    #     G_sub.add_edge(i, j)
                    if m[0]==n[0] and m[1]*n[1]:
                        G_sub.add_edge(i, j, attr=m[0])
                        G_sub_graph.add_edge(i, j, attr=m[0])
        # 单个节点的小子图处理
        for i, j in itertools.combinations(G_sub.nodes(), 2):
            if subgraphs[i].nodes() == 1 and subgraphs[j].nodes() == 1:
                if G.has_edge(subgraphs[i].nodes()[0], subgraphs[j].nodes()[0]):
                    G_sub.add_edge(i, j)
                    G_sub_graph.add_edge(i, j)
        # nx.draw(G_sub, with_labels=True)
        # plt.show()
        # A=self.has_cycle(self,G_sub)
        # print(A)
        return G_sub, G_sub_graph

    def domain_route(self, service, G_sub, subgraphs):
        all_path = []
        Flag = 1
        # 需要做OEO转换的service
        # src和snk在同一子图
        # for i, sg in enumerate(subgraphs):
        #     if sg.has_node(service['src']) and sg.has_node(service['snk']):
        #         # all_path = list(nx.all_simple_paths(sg, service['src'], service['src']))
        #         all_path = [[i], ]
        #         Flag = 0
        if self.G.nodes[service['src']]['domain'] & self.G.nodes[service['snk']]['domain']:
            all_path = [[i] for i in self.G.nodes[service['src']]['domain'] & self.G.nodes[service['snk']]['domain']]
            Flag = 0

        # src和snk在不同子图
        if Flag:
            find_path = False
            for s in self.G.nodes[service['src']]['domain']:
                if find_path:
                    break
                for d in self.G.nodes[service['snk']]['domain']:
                    if sum(i[1] for i in G_sub.nodes[s]['relay_node']) and sum(i[1] for i in G_sub.nodes[d]['relay_node']) \
                            and nx.has_path(G_sub, s, d):  # 如果子图中包含s的源节点且子图中所有中继节点的中继器之和>0
                        # all_possible_paths = list(nx.all_simple_paths(G_sub, s, d))
                        all_path = list(self.YenKSP(G_sub, s, d, K=5))
                        find_path = True
                        break

            # for s, d in itertools.combinations(G_sub.nodes(), 2):
            #     if subgraphs[s].has_node(service['src']) and sum(i[1] for i in G_sub.nodes[s]['relay_node']) \
            #             and subgraphs[d].has_node(service['snk']) and sum(i[1] for i in G_sub.nodes[d]['relay_node']) \
            #             and nx.has_path(G_sub, s, d):  # 如果子图中包含s的源节点且子图中所有中继节点的中继器之和>0
            #         # all_possible_paths = list(nx.all_simple_paths(G_sub, s, d))
            #         all_possible_paths = list(self.YenKSP(G_sub, s, d, K=10))
            #         for path in all_possible_paths:
            #             for hop in path:
            #                 if sum(r[1] for r in
            #                        G_sub.nodes[hop]['relay_node']) <= 0:  # 如果途径的某一子图中无中继器可以使用，则该路径作废，不计入all_path
            #                     break
            #             all_path.append(path)

        # 不需要做OEO转换的service？？
        # for s, d in itertools.combinations(G_sub.nodes(), 2):
        #     if subgraphs[s].has_node(service['src']) and subgraphs[s].nodes() == 1 \
        #             and subgraphs[d].has_node(service['snk']) and subgraphs[d].nodes() == 1:
        #         all_path = list(self.YenKSP(G_sub, s, d, K=10))

        # print(len(all_path))
        return all_path

    def YenKSP(self, G, source, target, K):
        path_list = []
        path_list.append(nx.dijkstra_path(G, source, target, weight='weight'))

        for k in range(K - 1):
            temp_path = []
            for i in range(len(path_list[k]) - 1):
                tempG = G.copy()  # 复制一份图 供删减操作
                spurNode = path_list[k][i]
                rootpath = path_list[k][:i + 1]
                len_rootpath = nx.dijkstra_path_length(tempG, source, spurNode, weight='weight')

                for p in path_list:
                    if rootpath == p[:i + 1]:
                        if tempG.has_edge(p[i], p[i + 1]):
                            tempG.remove_edge(p[i], p[i + 1])  # 防止与祖先状态重复
                tempG.remove_nodes_from(rootpath[:-1])  # 防止出现环路
                if not (nx.has_path(tempG, spurNode, target)):
                    continue  # 如果无法联通，跳过该偏移路径

                spurpath = nx.dijkstra_path(tempG, spurNode, target, weight='weight')
                len_spurpath = nx.dijkstra_path_length(tempG, spurNode, target, weight='weight')

                totalpath = rootpath[:-1] + spurpath
                len_totalpath = len_rootpath + len_spurpath
                temp_path.append([totalpath, len_totalpath])
            if len(temp_path) == 0:
                break

            temp_path.sort(key=(lambda x: [x[1], len(x[0])]))  # 按路程长度为第一关键字，节点数为第二关键字，升序排列
            path_list.append(temp_path[0][0])

        return path_list

    def indomain_route(self, subgraph_index, s, s_r, d, flag):  # flag: 0:source domain 1: dis domain 2: inter domain
        if flag == 0 or flag == 3:  # source domain
            f_slot_relay_s = [1 for i in range(self.L)]
        else:
            f_slot_relay_s = self.G.nodes[s]['available relay'][s_r]['f_slot']

        if flag == 2 or flag == 3:  # dis domain
            f_slot_relay_d = [1 for i in range(self.L)]
            for l in range(self.L):
                if f_slot_relay_s[l] and f_slot_relay_d[l]:
                    route = self.route(self.subgraphs_layering[subgraph_index][l], s, d)
                    edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                    if route != 0 and len(edge_index_of_path) != 0:
                        return route, l, -1, edge_index_of_path

        for d_r, relay in enumerate(self.G.nodes[d]['available relay']):
            if not relay['available']:
                continue
            f_slot_relay_d = relay['f_slot']
            for l in range(self.L):
                if f_slot_relay_s[l] and f_slot_relay_d[l]:
                    route = self.route(self.subgraphs_layering[subgraph_index][l], s, d)
                    edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                    if route != 0 and len(edge_index_of_path) != 0:
                        return route, l, d_r, edge_index_of_path
        return 0, -1, -1, []

    def if_spectrum_available_for_path(self, path, l):
        if path == 0:
            return []
        edge_index_of_path = []
        for i in range(len(path) - 1):
            for k, e in enumerate(self.G[path[i]][path[i+1]].values()):
                if e['f_slot'][l] == 1:
                    edge_index_of_path.append(k)
                    break
            if len(edge_index_of_path) <= i:  # 所有多重边不可用
                return []
        return edge_index_of_path

    def updata_spectrum(self, path, edge_index_of_path, l):
        for i in range(len(path) - 1):
            self.G[path[i]][path[i + 1]][edge_index_of_path[i]]['f_slot'][l] = 0

    def updata_state(self, domain_route, sub_routes):
        for r in sub_routes:
            self.updata_spectrum(r['route'], r['edge_index_of_path'], r['layer'])
            if r['relay_index'] != -1:
                self.G.nodes[r['route'][-1]]['available relay'][r['relay_index']]['available'] = False
        # update graph layering
        for d, domain in enumerate(domain_route):
            route = sub_routes[d]['route']
            edge_index_of_path = sub_routes[d]['edge_index_of_path']
            layer = sub_routes[d]['layer']
            for i in range(len(route) - 1):
                self.subgraphs_layering[domain][layer].remove_edge(route[i], route[i+1], edge_index_of_path[i])


    def run(self):
        # subgraphs = load_sg(path)
        # G_sub = nx.MultiGraph()
        G_sub, G_sub_graph = self.construct_G_sub(self.G, self.subgraphs)  # 每个子图在其中抽象为一个节点
        resource_occupation_before = self.resource_occupation(self.G)
        num_succeed = 0
        time_succeed_list = []
        time_indomain_list = []
        time_for_service = []  # 域间时间+最长域内时间
        len_domain_list = []
        start_total = time.time()
        a = 0
        for service in self.services:
            start_service = time.time()
            a += 1
            if a == 401:
                b=1
            print(len(self.services), a)
            # domains route (relay can be used)
            start_domain = time.time()
            domain_routes = self.domain_route(service, G_sub_graph, self.subgraphs)
            end_domain = time.time()
            time_domain = end_domain - start_domain
            print('domain routes:', domain_routes)
            # route in G Layering for each domain in each layer (First Fit)
            for domain_route in domain_routes:
                time_indomain_for_service = []
                sub_routes = []
                success_flag = 0
                if len(domain_route) == 1:
                    s = service['src']
                    d = service['snk']
                    subgraph_index = domain_route[0]
                    start_indomain = time.time()
                    route, l, d_r, edge_index_of_path = self.indomain_route(subgraph_index, s, 0, d, flag=3)
                    end_indomain = time.time()
                    time_indomain_list.append(end_indomain - start_indomain)
                    time_indomain_for_service.append(end_indomain - start_indomain)
                    # edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                    # print(route, edge_index_of_path)
                    if route != 0 and len(edge_index_of_path) != 0:
                        sub_routes.append({'route': route, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                        # success find route
                        success_flag = 1
                else:
                    for i, subgraph_index in enumerate(domain_route):
                        start_indomain = time.time()

                        # domain s
                        if i == 0:
                            indomain_success_flag = 0
                            # s_sub = [service['src']]
                            s = service['src']
                            d_sub = []
                            for e in G_sub[subgraph_index][domain_route[i + 1]].values():
                                d_sub.append(e['attr'])

                            for d in d_sub:
                                route, l, d_r, edge_index_of_path = self.indomain_route(subgraph_index, s, 0, d, flag=0)
                                # edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                                # print(route, edge_index_of_path)
                                if route != 0 and len(edge_index_of_path) != 0:
                                    sub_routes.append({'route': route, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                                    indomain_success_flag = 1
                                    break  # find indomain route, next domain
                            if indomain_success_flag:  # find indomain route, next domain
                                continue
                            else:
                                break  # can't find route, next domain route

                        # domain d
                        elif i == len(domain_route) - 1:
                            s = sub_routes[-1]['route'][-1]  # 上个域路由的最后节点
                            d = service['snk']
                            route, l, d_r, edge_index_of_path = self.indomain_route(subgraph_index, s, sub_routes[-1]['relay_index'], d, flag=2)
                            # edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                            # print(route, edge_index_of_path)
                            if route != 0 and len(edge_index_of_path) != 0:
                                sub_routes.append({'route': route, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                                # success find route
                                success_flag = 1

                        # domain inter
                        else:
                            # s_sub = []
                            # for e in G_sub[domain_route[i - 1]][subgraph_index].values():
                            #     s_sub.append(e['attr'])
                            indomain_success_flag = 0
                            s = sub_routes[-1]['route'][-1]  # 上个域路由的最后节点
                            d_sub = []
                            for e in G_sub[subgraph_index][domain_route[i + 1]].values():
                                d_sub.append(e['attr'])
                            for d in d_sub:
                                route, l, d_r, edge_index_of_path = self.indomain_route(subgraph_index, s, sub_routes[-1]['relay_index'], d, flag=1)
                                # edge_index_of_path = self.if_spectrum_available_for_path(route, l)
                                # print(route, edge_index_of_path)
                                if route != 0 and len(edge_index_of_path) != 0:
                                    sub_routes.append({'route': route, 'layer': l, 'relay_index': d_r, 'edge_index_of_path': edge_index_of_path})
                                    indomain_success_flag = 1
                                    break  # find indomain route, next domain
                            if indomain_success_flag:  # find indomain route, next domain
                                continue
                            else:
                                break  # can't find route, next domain route

                        end_indomain = time.time()
                        time_indomain_list.append(end_indomain-start_indomain)
                        time_indomain_for_service.append(end_indomain - start_indomain)
                        # for s in s_sub:
                        #     for d in d_sub:
                        #         if not self.G.nodes[s]['relay']:
                        #             f_slot_relay_s = [1 for i in range(self.L)]
                        #         else:
                        #
                        #         for l in range(self.L):
                        #             route = self.route(self.subgraphs_layering[subgraph_index][l], )

                if success_flag:
                    # update G, G Layering
                    # print('update:', domain_route, sub_routes)
                    self.updata_state(domain_route, sub_routes)
                    num_succeed += 1
                    end_service = time.time()
                    time_succeed_list.append(end_service-start_service)
                    time_for_service.append(time_domain + max(time_indomain_for_service))
                    # len_domain_list.append(len(domain_route))
                    len_domain_list.extend([len(r) for r in domain_routes])
                    print('success')
                    break
                # print(domain_route)


            # update G, G Layering


        end_total = time.time()
        time_total = end_total - start_total
        resource_occupation_after = self.resource_occupation(self.G)
        print('num_succeed:', num_succeed)
        print(f"Service success rate: {num_succeed/len(self.services)}%")
        print('ave time (success):', np.mean(time_succeed_list))
        print('time indomain:', np.mean(time_indomain_list))
        print('len domain route:', np.mean(len_domain_list))
        print('time for service:', np.mean(time_for_service))
        print(
            f"resource occupation before: {resource_occupation_before}, resource occupation after: {resource_occupation_after}")
        print(len_domain_list)

if __name__ == '__main__':
    distance_margin = 800
    ots_margin = 10
    osnr_margin = 0.01


    # file_name = 'example1'
    # # subgraph_file_path = './subgraphs/' + file_name + '/'
    # subgraph_file_path = './subgraphs/' + file_name + 'greedy' + '/'
    # S = ServiceRecovery(distance_margin, ots_margin, osnr_margin, subgraph_file_path=subgraph_file_path, file_path='example/', file_name='example1', band=24)

    file_name = 'example2'
    # subgraph_file_path = './subgraphs/' + file_name + '/'
    subgraph_file_path = './subgraphs/' + file_name + 'greedy' + '/'
    S = ServiceRecovery(distance_margin, ots_margin, osnr_margin, subgraph_file_path=subgraph_file_path, file_path='example/', file_name='example2', band=8)

    # path = './subgraphs/' + file_name + '/'
    # start_time = time.time()
    S.run()
    # end_time = time.time()
