import pandas as pd
import ast
import os

scriptpath = os.path.dirname(__file__)
# 读取CSV文件
df = pd.read_csv(scriptpath + '/../database/ETA/road_features.csv')

# 定义函数来提取坐标
def extract_coordinates(row):
    return ast.literal_eval(row["coordinates"])

# 应用函数，提取坐标列
df["coordinates"] = df.apply(extract_coordinates, axis=1)

# 提取标签和特征
labels = df["id"].astype(int)
coordinates = df["coordinates"]
features = df.iloc[:, 2:].values.astype(float)

# 打印结果
print("Labels:", labels.values)
print("Coordinates:", coordinates.values)
print("Features:\n", features)
