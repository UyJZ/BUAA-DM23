import pandas as pd
import networkx as nx
import os
import ast
import numpy as np
from shapely.geometry import Point, LineString
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from tqdm import tqdm
from rtree import index as rtree_index

script_dir = os.path.dirname(os.path.realpath(__file__))
graphPath = os.path.join(script_dir, 'data', 'road_graph.gpickle')
df = pd.read_csv(os.path.join(script_dir, 'data', 'road.csv'))
traj = pd.read_csv(os.path.join(script_dir, 'data', 'traj.csv'))

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

coordinate_to_road_id = {}
for index, row in df.iterrows():
    coordinates = np.array(eval(row['coordinates']))
    for i in range(len(coordinates) - 1):
        point1, point2 = coordinates[i], coordinates[i + 1]
        key = (point1[0], point1[1], point2[0], point2[1])
        coordinate_to_road_id[key] = row['id']

def getRoadID(from_x, from_y, to_x, to_y):
    key = (from_x, from_y, to_x, to_y)
    return coordinate_to_road_id.get(key, -1)

def find_closest_edge_1(graph, coordinates):
    # 创建空间索引
    idx = rtree_index.Index()
    for i, (edge_start, edge_end, data) in enumerate(graph.edges(data=True)):
        start_coord, end_coord = np.array(edge_start), np.array(edge_end)
        idx.insert(i, (min(start_coord[0], end_coord[0]), min(start_coord[1], end_coord[1],
                                                               max(start_coord[0], end_coord[0]),
                                                               max(start_coord[1], end_coord[1]))))
    
    # 处理每个点
    results = []
    for coordinate in coordinates:
        point = Point(coordinate)
        min_distance = float('inf')
        closest_edge = None
        
        # 查询空间索引以获取潜在的边
        for i in idx.intersection((coordinate[0], coordinate[1], coordinate[0], coordinate[1])):
            edge_start, edge_end, data = graph.edges(data=True)[i]
            start_coord, end_coord = np.array(edge_start), np.array(edge_end)
            distance = LineString([start_coord, end_coord]).distance(point)
            
            if distance < min_distance:
                min_distance = distance
                closest_edge = (edge_start, edge_end)
        
        if closest_edge is not None:
            results.append(getRoadID(*closest_edge[0], *closest_edge[1]))
        else:
            results.append(-1)
    
    return results

def find_closest_edge(graph, coordinate):
    point = Point(coordinate)
    closest_edge = None
    min_distance = float('inf')  # Initialize with infinity
    
    for edge_start, edge_end, data in graph.edges(data=True):
        edge_coordinates = (edge_start, edge_end)
        
        # Extract the coordinates from the nodes
        start_coord, end_coord = np.array(edge_start), np.array(edge_end)
        
        # Calculate the distance from the point to the edge
        distance = LineString([start_coord, end_coord]).distance(point)
        
        if distance < min_distance:
            min_distance = distance
            closest_edge = edge_coordinates
    
    if closest_edge is not None:
        return getRoadID(*closest_edge[0], *closest_edge[1])  # Call getRoadID method to return the edge ID
    
    return -1  # If not close to any edge, return -1 or an appropriate value

def find_edge_for_trajectory_parallel(graph, trajectory):
    results = []

    with tqdm(total=len(trajectory)) as pbar:  # 使用 tqdm 创建进度条
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(find_closest_edge, graph, ast.literal_eval(row['coordinates'])) for _, row in trajectory.iterrows()]
            
            for future in concurrent.futures.as_completed(futures):
                road_id = future.result()
                results.append({'road_id': road_id})
                pbar.update(1)  # 更新进度条
                print('Finished processing road_id: {}'.format(road_id))
    
    result_df = pd.DataFrame(results)
    result_df['coordinate'] = trajectory['coordinates'].values
    return result_df

# 将 roadId 添加到轨迹并保存到新的 CSV 文件
result_df_parallel = find_edge_for_trajectory_parallel(graph, traj)
traj['roadId'] = result_df_parallel['road_id'].values
traj.to_csv(os.path.join(script_dir, 'data', 'traj_roadId.csv'), index=False)
