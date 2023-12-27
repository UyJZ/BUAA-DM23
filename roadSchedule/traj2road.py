import pandas as pd
import os
import ast
import geopandas as gpd
import osmnx as ox
import networkx as nx
import transbigdata
from shapely.geometry import Point
from path import *

# 获取当前脚本的路径
df = pd.read_csv(os.path.join(script_dir, 'database', 'data', 'road.csv'))

# 构建Ox图
def buildOxGraph(graph_path, df):
    if os.path.exists(graph_path):
        return

    G = nx.DiGraph()

    for index, row in df.iterrows():
        coordinates = eval(row['coordinates'])
        for i in range(len(coordinates) - 1):
            point1, point2 = coordinates[i], coordinates[i + 1]
            distance = ox.distance.great_circle_vec(point1[0], point1[1], point2[0], point2[1])

            G.add_node(tuple(point1), x=point1[0], y=point1[1])
            G.add_node(tuple(point2), x=point2[0], y=point2[1])

            G.add_edge(tuple(point1), tuple(point2), length=distance)

    # 为图分配坐标参考系统
    G.graph["crs"] = "EPSG:4326"

    nx.write_gpickle(G, graph_path)

# 使用已有的数据构建图
buildOxGraph(graphPath, df)

# 获取图
def getGraph(graph_path):
    if os.path.exists(graph_path):
        return nx.read_gpickle(graph_path)
    else:
        buildOxGraph(graph_path, df)
        return nx.read_gpickle(graph_path)

# 读取轨迹数据集
traj_data = pd.read_csv(os.path.join(script_dir, 'database', 'data', 'traj.csv'))

# 添加 'lon' 和 'lat' 列
traj_data['lon'] = traj_data['coordinates'].apply(lambda x: ast.literal_eval(x)[0])
traj_data['lat'] = traj_data['coordinates'].apply(lambda x: ast.literal_eval(x)[1])

# 读取路网图
G = getGraph(graphPath)


# 获取节点和边的GeoDataFrames

gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, node_geometry=True, fill_edge_geometry=True)

# 进行轨迹到路径的匹配
matched_traj = transbigdata.traj_mapmatch(traj_data.head(5).copy(), G, col=['lon', 'lat'])

# 输出匹配后的轨迹数据集
print(matched_traj)
