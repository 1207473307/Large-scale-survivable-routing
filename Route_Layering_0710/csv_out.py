#csv_out
import csv
import networkx as nx
# 创建一个空的无向图
G = nx.Graph()
# CSV文件路径
ex='./example/example1.'
file_node =ex+ 'node.csv'
file_edge =ex+ 'oms.csv'
file_relay=ex+ 'relay.csv'
file_service=ex+ 'service.csv'

def Get_G():
    # 使用csv模块打开CSV文件
    with open(file_node, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            node=row[1]
            G.add_node(node)


    with open(file_edge, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        # 跳过标题行
        h=next(csvreader)
        # 读取CSV文件中的每一行
        for row in csvreader:
            # 假设第一列是源节点，第二列是目标节点
            source_node = int(row[h.index('src')])
            target_node = int(row[h.index('snk')])
            cost=float(row[h.index('cost')]) 
            distance=float(row[h.index('distance')])
            ots=float(row[h.index('ots')])
            osnr=float(row[h.index('osnr')])
            colors=(row[h.index('colors')])
            parts = colors.split(':')
            colors = [list(map(int, part.split('-'))) for part in parts if part]
            # 将节点和边添加到图中
            if colors!='':
                G.add_edge(source_node,target_node,attr_dict={'cost':cost,'distance':distance,'ots':ots,'osnr_loss':osnr})


    relay_list=[]
    with open(file_relay, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            relay_node = row[3]
            if relay_node in row:
                # 如果节点不存在，则添加节点
                if not G.has_node(relay_node):
                    G.add_node(relay_node)
                # 修改节点属性为1
                G.nodes[relay_node]['relay'] = 1
                relay_list.append(int(relay_node))

    #添加service信息
    service=[]
    with open(file_service, mode='r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            s=float(row[1])
            d=float(row[2])
            m_width=float(row[5])
            s_dim=row[7]
            parts = s_dim.split(':')
            #波段处理
            s_ranges = [list(map(int, part.split('-'))) for part in parts if part]
            for item in s_ranges:
                if item[1]-item[0]<m_width:
                    s_ranges.remove(item)
            d_dim=row[8]
            parts = d_dim.split(':')
            d_ranges = [list(map(int, part.split('-'))) for part in parts if part]
            for item in d_ranges:
                if item[1]-item[0]<m_width:
                    d_ranges.remove(item)
            service.append([s,d,s_ranges,d_ranges])
        
    return G,relay_list,service