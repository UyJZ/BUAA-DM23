import os
from path import *
import networkx as nx
import pandas as pd
import numpy as np


df = pd.read_csv(os.path.join(script_dir, 'database', 'data', 'road.csv'))

d = {}

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

def Dijkstra(from_x, from_y, to_x, to_y):
    from_node = min(graph.nodes(), key=lambda node: (node[0] - from_x) ** 2 + (node[1] - from_y) ** 2)
    to_node = min(graph.nodes(), key=lambda node: (node[0] - to_x) ** 2 + (node[1] - to_y) ** 2)
    path = nx.shortest_path(graph, source=from_node, target=to_node, weight='length')
    path_coordinates = [(node[0], node[1]) for node in path]
    road_ids = []
    weights = []

    for i in range(len(path) - 1):
        from_coord = (path[i][0], path[i][1])
        to_coord = (path[i + 1][0], path[i + 1][1])
        road_id = getRoadID(from_coord[0], from_coord[1], to_coord[0], to_coord[1])
        if road_id != -1:
            road_ids.append(road_id)
            weights.append(graph[from_coord][to_coord]['length'])

    return path_coordinates, road_ids, weights

coordinate_to_road_id = {}
for index, row in df.iterrows():
    coordinates = np.array(eval(row['coordinates']))
    for i in range(len(coordinates) - 1):
        point1, point2 = coordinates[i], coordinates[i + 1]
        key = (point1[0], point1[1], point2[0], point2[1])
        coordinate_to_road_id[key] = row['id']
        d[key] = row['id']

def getRoadID(from_x, from_y, to_x, to_y):
    key = (from_x, from_y, to_x, to_y)
    return coordinate_to_road_id.get(key, -1)

def getOriginRoadId(from_x, from_y, to_x, to_y):
    key = (from_x, from_y, to_x, to_y)
    return d.get(key, -1)

def getOriginRoadIdByNewRoadId(new_road_id):
    for key, value in coordinate_to_road_id.items():
        if value == new_road_id:
            return d.get(key, -1)