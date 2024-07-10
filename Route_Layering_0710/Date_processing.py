import sys
import numpy
import csv
import networkx as nx
import matplotlib.pyplot as plt

# The colors data is processed as f_slot, and n colors are merged into one band   # 将colors数据处理为频隙，n个color合并为一个波段
def process_colors(colors, n, c_max=964):
    N = int(c_max/n)
    f_slot = [0 for i in range(N)]
    segments = colors.split(':')
    while segments:
        s = segments.pop(0)
        if s == '':
            continue
        start, end = map(int, s.split('-'))
        for j in range(int(start/n), int(end/n)):
            f_slot[j] = 1

    return f_slot


def create_topology(file_path, file_name, band=24, c_max=964):
    network = nx.MultiGraph()  # multigraph  # 多重无向图，运行平行边
    network.graph['L'] = int(c_max/band)
    # Read the node.csv file and add nodes to the graph  # 读取节点文件，向图中添加节点
    with open(file_path+file_name+'.node.csv') as f1:
        Node = csv.DictReader(f1)
        node_list = []
        for n in Node:
            node_list.append((int(n['nodeId']), {'relay': False, 'available relay num': 0, 'available relay': []}))
            # network.add_node(n['nodeId'], 'relay'=False)
        # network.add_nodes_from(node_list, relay=False, available_relay_num=0, available_relay=[])
        network.add_nodes_from(node_list)

    # Read the oms.csv file and add edges to the graph  # 读取oms文件，向图中添加边
    with open(file_path+file_name+'.oms.csv') as f2:
        Edge = csv.DictReader(f2)
        edge_list = []
        for e in Edge:
            edge_list.append((int(e['src']), int(e['snk']),
                              {'omsId': int(e['omsId']),
                               'remoteOmsId': int(e['remoteOmsId']),
                               'cost': float(e['cost']),
                               'distance': float(e['distance']),
                               'ots': float(e['ots']),
                               'osnr': float(e['osnr']),
                               'f_slot': process_colors(e['colors'], n=band)
                               }))
        network.add_edges_from(edge_list)
        # print(edge_list)

    # Process the relay.csv file to add available relay attributes to the node  # 处理relay文件，向节点添加可用中继属性
    with open(file_path+file_name+'.relay.csv') as f3:
        Relay = csv.DictReader(f3)
        for r in Relay:
            network.nodes[int(r['nodeId'])]['relay'] = True
            network.nodes[int(r['nodeId'])]['available relay num'] += 1
            network.nodes[int(r['nodeId'])]['available relay'].append(
                {'available': True,
                 'relayId': int(r['relayId']),
                 'relatedRelayId': int(r['relatedRelayId']),
                 'localId': int(r['localId']),
                 'relatedLocalId': int(r['relatedLocalId']),
                 'f_slot': process_colors(r['dimColors'], n=band)
                 })
    # nx.draw(network, with_labels=True)
    # plt.show()
    return network


def process_service(file_path, file_name, band=24):
    service_list = []
    with open(file_path+file_name+'.service.csv') as f1:
        Service = csv.DictReader(f1)
        for s in Service:
            service_list.append(
                {'Index': s['Index'],
                 'src': int(s['src']),
                 'snk': int(s['snk']),
                 'sourceOtu': int(s['sourceOtu']),
                 'targetOtu': int(s['targetOtu']),
                 'm_width': s['m_width'],
                 'bandType': s['bandType'],
                 'sourceDimf': process_colors(s['sourceDimColors'], n=band),
                 'targetDimf': process_colors(s['targetDimColors'], n=band),
                 }
            )
    # print(service_list)
    return service_list


def get_relay_node(G):
    relay_node = []
    for n in G.nodes():
        if G.nodes[n]['relay']:
            relay_node.append(n)
    return relay_node


if __name__ == "__main__":
    G = create_topology(file_path='example/', file_name='example1')
    # process_service(file_path='example/', file_name='example1')
    relay_node = get_relay_node(G)
    print('number of relay nodes:', len(relay_node))
    print('number of nodes:', G.number_of_nodes())