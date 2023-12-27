from path import *
import pandas as pd
from ast import literal_eval
import os
import networkx as nx
from tensorflow import keras

df = pd.read_csv(os.path.join(script_dir, 'database', 'data', 'road.csv'))

def buildGraph(graph_path):
    if os.path.exists(graph_path):
        return
    G = nx.DiGraph()
    for index, row in df.iterrows():
        coordinates = eval(row['coordinates'])
        for i in range(len(coordinates) - 1):
            point1, point2 = coordinates[i], coordinates[i + 1]
            distance = row['length'] / (len(coordinates) - 1)
            G.add_edge(tuple(point1), tuple(point2), length=distance)
    nx.write_gpickle(G, graph_path)

def getGraph(graph_path):
    if os.path.exists(graph_path):
        return nx.read_gpickle(graph_path)
    else:
        buildGraph(graph_path)
        return nx.read_gpickle(graph_path)
    
graph = getGraph(graph_path=graphPath)
print('num of edges : ', graph.number_of_edges())

df = pd.read_csv(fmm_path, delimiter=';')

df['cpath'].fillna('[]', inplace=True)

# 将cpath列中的字符串列表转换为实际的Python列表
df['cpath'] = df['cpath'].apply(lambda x: literal_eval(x) if pd.notna(x) else [])

# 使用set获取所有元素的总种类
# 使用set获取所有元素的总种类
all_elements = set()
for sublist in df['cpath']:
    if isinstance(sublist, (list, tuple)):
        all_elements.update(sublist)
    else:
        all_elements.add(sublist)

# 打印结果
print("总种类数:", len(all_elements))
df1 = pd.read_csv(train_set_path_holiday_0, index_col=0)