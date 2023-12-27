import pandas as pd
import ast, os
from path import *
script_dir = os.path.dirname(os.path.realpath(__file__))
# 读取CSV文件
df = pd.read_csv(script_dir + '/database/data/eta_task.csv')
def extract_coordinates(coord_str):
    return ast.literal_eval(coord_str)

# 提取起点和终点的坐标
df['from'] = df['coordinates'].apply(lambda x: extract_coordinates(x) if pd.notnull(x) else None)
df['to'] = df['coordinates'].shift(-1).apply(lambda x: extract_coordinates(x) if pd.notnull(x) else None)

# 提取速度信息
df['speed_src'] = df['speeds']
df['speed_dst'] = df['speeds'].shift(-1)

# 只保留一个'holidays'列
df['holidays'] = df['holidays'].fillna(method='ffill')
df['holidays_dst'] = df['holidays'].shift(-1)

# 删除多余的列
df = df.drop(['coordinates', 'current_dis', 'speeds'], axis=1)

# 保留第2k-1行，删除第2k行
result_df = df.iloc[::2]
result_df.to_csv(script_dir + '/database/data/result_eta_task.csv', index=False)