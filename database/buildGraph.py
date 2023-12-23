import pandas as pd
import networkx as nx
import os
import ast
from rtree import index
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

script_dir = os.path.dirname(os.path.realpath(__file__))
graphPath = os.path.join(script_dir, 'data', 'road_graph.gpickle')
df = pd.read_csv(os.path.join(script_dir, 'data', 'road.csv'))

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

def getRoadID(from_x, from_y, to_x, to_y):
    key = (from_x, from_y, to_x, to_y)
    return coordinate_to_road_id.get(key, -1)

def process_single_trajectory(from_coord, to_coord, traj_id):
    global result_data
    global noPath
    try:
        path, roads, weights = Dijkstra(from_coord[0], from_coord[1], to_coord[0], to_coord[1])
        result_data['t'].append(traj_id)
        result_data['path'].append(path)
        result_data['roads'].append(roads)
        result_data['weights'].append(weights)
    except Exception as e:
        print(e)
        noPath += 1
        result_data['t'].append(traj_id)
        result_data['path'].append(None)
        result_data['roads'].append(None)
        result_data['weights'].append(None)

def main():
    global result_data
    global noPath
    buildGraph(graphPath)

    result_df = pd.read_csv(os.path.join(script_dir, 'data', 'result_eta_task.csv'))
    from_coordinates = result_df['from'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)
    to_coordinates = result_df['to'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else None)

    result_data = {
        't': [],
        'path': [],
        'roads': [],
        'weights': []
    }

    noPath = 0

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_trajectory, from_coord, to_coord, traj_id)
                   for from_coord, to_coord, traj_id in zip(from_coordinates, to_coordinates, result_df['traj_id'])]
        for future in tqdm(futures, total=len(futures), desc="Processing trajectories"):
            future.result()

    print(f"No path found for {noPath} cases.")

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(os.path.join(script_dir, 'data', 'eta_task_schedule.csv'), index=False)

if __name__ == "__main__":
    main()
