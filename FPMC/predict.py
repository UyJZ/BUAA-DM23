import os
from utils import load_jump_task_from
from FPMC import FPMC
import pandas as pd
import numpy as np

cwd = os.path.dirname(os.path.realpath(__file__))

data_list = load_jump_task_from(cwd + '/data/jump.txt')



model = (FPMC.load(cwd + '/model.pkl'))

print('load successfully')


allowed_trans = {}

n_road = 38026

df = pd.read_csv(cwd + '/../database/data/rel.csv')

for index, row in df.iterrows():
    origin_id = int(row['origin_id'])
    destination_id = int(row['destination_id'])
    if origin_id not in allowed_trans:
        allowed_trans[origin_id] = [origin_id]
    if destination_id != origin_id and destination_id not in allowed_trans[origin_id]:
        allowed_trans[origin_id].append(destination_id)
for i in range(n_road + 1):
    if i not in allowed_trans:
        allowed_trans[i] = [i]
        
num = len(data_list) # 预测的数量默认为data_list的总数量，想要少点的话可以自己改

for i in range(num):
    l = data_list[i]
    u, i, b_tm1 = l
    current_status = b_tm1[-1]
    res = -np.inf
    best_path = []
    print('current_status : .{}'.format(current_status))
    for best_choice in allowed_trans[current_status]:
        if best_choice == b_tm1[-1]:
            continue
        road = list(b_tm1)
        #road.append(best_choice)
        r = model.compute_x(u=u, b_tm1=road, i=best_choice)
        if r > res:
            road.append(best_choice)
            best_path = road
            res = r
    print('最佳预测为.{}'.format(best_path[-1]))
    print('路径为: .{} '.format(best_path))
        
        