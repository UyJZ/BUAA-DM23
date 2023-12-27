import pandas as pd
import networkx as nx
import os
import ast
import numpy as np
from path import *
from Graph import *

def convert_coordinates(coordinates_str):
    return ast.literal_eval(coordinates_str)

def getOriginRoadIds(coordinates):
    road_ids = []
    for i in range(len(coordinates) - 1):
        from_coord = coordinates[i]
        to_coord = coordinates[i + 1]
        key = (from_coord[0], from_coord[1], to_coord[0], to_coord[1])
        road_id = d.get(key, -1)
        if road_id != -1:
            road_ids.append(road_id)
    return road_ids



fmm_df = pd.read_csv(fmm_out_path, delimiter=',')

# 重置索引以将 'coordinates' 列变为常规列
fmm_df = fmm_df.reset_index()
fmm_df['coordinates'] = fmm_df['coordinates'].apply(ast.literal_eval)
# 添加 'RoadIDs' 列
df['coordinates'] = df['coordinates'].apply(convert_coordinates)

fmm_df['RoadIDs'] = fmm_df['coordinates'].apply(getOriginRoadIds)
fmm_df = fmm_df.rename(columns={'coordinates': 'road_coordinates'})

traj_df = pd.read_csv(grouped_traj_path)
traj_df = traj_df.rename(columns={'coordinates': 'point_coordinates'})
merged_df = pd.merge(fmm_df, traj_df, on='id')

merged_df.to_csv(merged_output_path, index=False)

trainset_0 = merged_df[merged_df['holidays'] == 0]
trainset_1 = merged_df[merged_df['holidays'] == 1]
trainset_0.to_csv(train_set_path_holiday_0, index=False)
trainset_1.to_csv(train_set_path_holiday_1, index=False)

speeds_holidays_df = merged_df

speeds_holidays_df_exploded = speeds_holidays_df.explode('RoadIDs')

# 将 'RoadIDs' 列转为整数类型# 去除包含 NaN 值的行
speeds_holidays_df_exploded = speeds_holidays_df_exploded.dropna(subset=['RoadIDs'])

# 将 'RoadIDs' 列转为整数类型
speeds_holidays_df_exploded['RoadIDs'] = speeds_holidays_df_exploded['RoadIDs'].astype(int)


result_df = speeds_holidays_df_exploded.groupby(['RoadIDs', 'holidays'])['speeds'].agg(list).reset_index()

# 分别保存 'holidays' 为 0 和 1 的结果
output_df_holiday_0 = result_df[result_df['holidays'] == 0][['RoadIDs', 'speeds']]
output_df_holiday_1 = result_df[result_df['holidays'] == 1][['RoadIDs', 'speeds']]

# 保存结果
output_df_holiday_0.to_csv(output_path_holiday_0, index=False)
output_df_holiday_1.to_csv(output_path_holiday_1, index=False)

# 保存 result_df
result_df.to_csv(result_output_path, index=False)

# 输出路径
print(f"结果保存至 {output_path_holiday_0}, {output_path_holiday_1} 和 {result_output_path}")
