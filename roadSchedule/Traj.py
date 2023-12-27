import pandas as pd
import ast
import os
from path import *

# 读取原始CSV文件
df = pd.read_csv(traj_path)

# 将坐标列转换为列表
df['coordinates'] = df['coordinates'].apply(ast.literal_eval)

# 按照 traj_id 汇总速度信息和保留所有坐标
grouped_df = df.groupby('traj_id').apply(lambda x: pd.Series({
    'id': x['traj_id'].iloc[0],  # 设置 id 为 traj_id
    'speeds': x['speeds'].tolist(),  # 保留速度列表
    'holidays': x['holidays'].iloc[0],  # 保留第一个holidays值
    'coordinates': x['coordinates'].tolist(),  # 保留所有坐标
    # 在这里添加其他属性的处理
}))

# 将 traj_id 设置为新的主键
grouped_df.set_index('id', inplace=True)

# 保存结果到新的CSV文件
grouped_df.to_csv(grouped_traj_output_path)
