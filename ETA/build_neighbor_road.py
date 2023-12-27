import pandas as pd
from shapely.geometry import LineString, Polygon
from tqdm import tqdm
from path import road_path, grid_assignment_path, neighboring_roads_path

def create_grid(min_x, max_x, min_y, max_y, grid_size):
    grid = []
    x = min_x
    while x < max_x:
        y = min_y
        while y < max_y:
            grid.append(Polygon([(x, y), (x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]))
            y += grid_size
        x += grid_size
    return grid

def assign_roads_to_grid(roads, grid):
    assigned_roads = {i: [] for i in range(len(grid))}
    
    for idx, road in tqdm(roads.iterrows(), total=len(roads), desc="分配道路"):
        line = LineString(eval(road['coordinates']))
        
        for i, grid_polygon in enumerate(grid):
            if line.intersects(grid_polygon):
                assigned_roads[i].append(road['id'])
                
    return assigned_roads

def collect_neighboring_roads(assigned_roads):
    neighboring_roads = {}

    for grid_id, road_ids in tqdm(assigned_roads.items(), desc="收集相邻道路"):
        for road_id in road_ids:
            if road_id not in neighboring_roads:
                neighboring_roads[road_id] = set()
            neighboring_roads[road_id].update(set(road_ids) - {road_id})

    return neighboring_roads

df = pd.read_csv(road_path)

min_x, max_x = df['coordinates'].apply(lambda x: eval(x)[0][0]).min(), df['coordinates'].apply(lambda x: eval(x)[0][0]).max()
min_y, max_y = df['coordinates'].apply(lambda x: eval(x)[0][1]).min(), df['coordinates'].apply(lambda x: eval(x)[0][1]).max()
grid_size = 0.01 
grid = create_grid(min_x, max_x, min_y, max_y, grid_size)
assigned_roads = assign_roads_to_grid(df, grid)

neighboring_roads = collect_neighboring_roads(assigned_roads)

output_df_1 = pd.DataFrame(assigned_roads.items(), columns=['grid_id', 'road_ids'])
output_df_1.to_csv(grid_assignment_path, index=False)

neighboring_records = []
for road_id, neighboring_set in neighboring_roads.items():
    neighboring_records.append({'road_id': road_id, 'neighboring_road_ids': list(neighboring_set)})

output_df_2 = pd.DataFrame(neighboring_records)
output_df_2.to_csv(neighboring_roads_path, index=False)
