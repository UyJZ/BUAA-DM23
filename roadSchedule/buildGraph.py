import pandas as pd
import networkx as nx
import os
import ast
from rtree import index
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .path import *
from Graph import *


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
