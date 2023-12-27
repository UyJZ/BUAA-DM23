import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from ast import literal_eval
from path import train_set_path_holiday_0, script_dir, model_holiday_0_path, model_holiday_1_path, train_set_path_holiday_1, predict_path
import os

# 读取数据
df = pd.read_csv(train_set_path_holiday_0, index_col=0)

# 提取特征和目标变量
df['point_coordinates'] = df['point_coordinates'].apply(literal_eval)  # 将字符串转换为列表
X = np.array(df['point_coordinates'].apply(lambda x: x[0]).tolist())  # 仅使用第一个坐标以简化
y = np.array(df['speeds'].apply(lambda x: literal_eval(x)[0]))  # 将每个列表的第一个元素转换为浮点数

# 将数组重新形状为两列（经度和纬度）
X = X.reshape(-1, 2)

# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X, y)

# 使用模型预测新的 road_coordinates 对应的速度
new_road_coordinates = np.array(df['road_coordinates'].apply(literal_eval).apply(lambda x: x[0]).tolist())
new_road_coordinates = new_road_coordinates.reshape(-1, 2)
new_speed_pred = rf_model.predict(new_road_coordinates)

# 获取每个路段的 RoadIDs
road_ids_list = df['RoadIDs'].apply(literal_eval)

# 将预测结果填入对应的 RoadIDs
speeds_by_road_id = []
for road_ids, speeds in zip(road_ids_list, np.array_split(new_speed_pred, np.cumsum(df['RoadIDs'].apply(len))[:-1])):
    for road_id, speed in zip(road_ids, speeds):
        speeds_by_road_id.append({'RoadID': road_id, 'PredictedSpeed': speed})

# 将填入速度后的结果转换为 DataFrame
predictions = pd.DataFrame(speeds_by_road_id)

# 对每个 RoadID 填入的速度取平均值
average_speeds = predictions.groupby('RoadID')['PredictedSpeed'].mean().reset_index()

# 将预测结果保存为 CSV 文件
average_speeds.to_csv(predict_path, index=False)

# 打印平均速度结果
print(average_speeds)

import pickle

# 保存模型
with open(model_holiday_0_path, 'wb') as model_file:
    pickle.dump(rf_model, model_file)
    
    
# 读取数据
df = pd.read_csv(train_set_path_holiday_1, index_col=0)

# 提取特征和目标变量
df['point_coordinates'] = df['point_coordinates'].apply(literal_eval)  # 将字符串转换为列表
X = np.array(df['point_coordinates'].apply(lambda x: x[0]).tolist())  # 仅使用第一个坐标以简化
y = np.array(df['speeds'].apply(lambda x: literal_eval(x)[0]))  # 将每个列表的第一个元素转换为浮点数

# 将数组重新形状为两列（经度和纬度）
X = X.reshape(-1, 2)

# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X, y)

# 使用模型预测新的 road_coordinates 对应的速度
new_road_coordinates = np.array(df['road_coordinates'].apply(literal_eval).apply(lambda x: x[0]).tolist())
new_road_coordinates = new_road_coordinates.reshape(-1, 2)
new_speed_pred = rf_model.predict(new_road_coordinates)

# 获取每个路段的 RoadIDs
road_ids_list = df['RoadIDs'].apply(literal_eval)

# 将预测结果填入对应的 RoadIDs
speeds_by_road_id = []
for road_ids, speeds in zip(road_ids_list, np.array_split(new_speed_pred, np.cumsum(df['RoadIDs'].apply(len))[:-1])):
    for road_id, speed in zip(road_ids, speeds):
        speeds_by_road_id.append({'RoadID': road_id, 'PredictedSpeed': speed})

# 将填入速度后的结果转换为 DataFrame
predictions = pd.DataFrame(speeds_by_road_id)

# 对每个 RoadID 填入的速度取平均值
average_speeds = predictions.groupby('RoadID')['PredictedSpeed'].mean().reset_index()

# 将预测结果保存为 CSV 文件
average_speeds.to_csv(predict_path, index=False)

# 打印平均速度结果
print(average_speeds)

import pickle

# 保存模型
with open(model_holiday_1_path, 'wb') as model_file:
    pickle.dump(rf_model, model_file)