import os
from utils import load_jump_task_from
from FPMC import FPMC
import pandas as pd
import numpy as np
import csv
import json

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


predict_data = {} # (entity_id,traj_id) => (current_status,predict_status,prev_status)
road_ids = [] # 之后速度预测器的输入
traj_id2idx = {} # 与road_ids配套使用 traj_id => traj_id在road_idx中的index
ll_data = {} # (entity_id,traj_id) => 路段数
ll = 0
with open("predict.txt","w") as f:
    with open("predict_task.txt","w") as f2:
        for i in range(num):
            l = data_list[i]
            u, b_tm1, traj_id = l
            current_status = b_tm1[-1]
            res = -np.inf
            best_path = []
            score = model.compute_x_batch(u, b_tm1)
            # print('current_status : .{}'.format(current_status))
            # print('current path : ', b_tm1)
            # if current_status == 24123:
            #     print('here')
            best_path = list(b_tm1)
            best_path.append(current_status)
            for best_choice in allowed_trans[current_status]:
                if best_choice == b_tm1[-1]:
                    continue
                road = list(b_tm1)
                #road.append(best_choice)
                r = score[best_choice]
                if r > res:
                    road.append(best_choice)
                    best_path = road
                    res = r
            prev_status = best_path[-3]
            predict_data[(u,traj_id)] = (current_status,best_path[-1],prev_status)
            road_ids.append(best_path)
            traj_id2idx[traj_id] = i
            if len(best_path) > ll :
                ll = len(best_path) 
            ll_data[(u,traj_id)] = len(best_path)
            f.write('entity_id: ' + str(u) + ' traj_id: ' + str(traj_id) + ' 路径为: .{} '.format(best_path))
            f.write('\n')
            f2.write(str(u) + ' ' + str(traj_id) + ' ' + str(best_path[-1]) + '\n')
        f2.close()
    f.close()

print("finish predict")





# --------------------------------------------开始面向过程-----------------------------------------
# 根据road_id找坐标和distance
    
# 处理road.csv中的坐标输入
def str2Coordinates(input):
    input = input.replace('[',' ')
    input = input.replace(']',' ')
    input = input.replace(',',' ')
    str = input.split()
    result = []
    for i in range(0,len(str),2):
        x = float(str[i])
        y = float(str[i + 1])
        result.append((x,y))
    return result

def parseCooridinate(input):
    input = input.replace('[',' ')
    input = input.replace(']',' ')
    input = input.replace(',',' ')
    str = input.split()
    x = float(str[0])
    y = float(str[1])
    return (x,y)


# 计算“距离”
def calDis(c1,c2):
    x1,y1 = c1
    x2,y2 = c2
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)

# 返回两条线段的交点 c1-c2 c3-c4
def getCross(c1,c2,c3,c4) :
    d1_3 = calDis(c1,c3)
    d1_4 = calDis(c1,c4)
    d2_3 = calDis(c2,c3)
    d2_4 = calDis(c2,c4)
    if d1_3 < d2_3 and d1_3 < d2_4:
        return c1
    if d1_4 < d2_3 and d1_4 < d2_4:
        return c1
    return c2


from datetime import datetime

# 计算时间差,返回小时
def time_difference(date1, date2):
    # 将日期字符串转换为datetime对象  
    dt1 = datetime.strptime(date1, "%Y-%m-%dT%H:%M:%SZ")
    dt2 = datetime.strptime(date2, "%Y-%m-%dT%H:%M:%SZ")

    # 计算时间差  
    diff = dt2 - dt1

    # 返回小时
    total_hours = diff.days * 24 + diff.seconds / 3600
    return total_hours

def getHour(date):
    dt = datetime.strptime(date,"%Y-%m-%dT%H:%M:%SZ")
    return dt.hour


# 导入road信息
print("start load road.csv")
road_data = {} # road_id => (coordinates,length)
with open(cwd + '/../database/data/road.csv','r') as file :
    csv_reader = csv.reader(file)
    head = 0
    for row in csv_reader:
        if head == 0 :
            head = 1
            continue
        road_id = int(row[0])
        coordinates = str2Coordinates(row[1])
        length = float(row[3])
        road_data[road_id] = (coordinates,length)
    file.close()

# 给出路段，坐标，返回到该路段剩余的距离
# TODO 我这里摆了，把他当直线处理的
def getRemainDis(road_id,cross,coordinate): 
    coordinates,length = road_data[road_id]
    # 正南正北
    if abs(coordinates[0][0] - coordinates[-1][0]) * 1000 < 1 :
        if abs(coordinates[0][1] - coordinates[-1][1]) * 1000 < 1 :
            return 0
        else :
            return abs(length * (1000 * cross[1] - 1000 * coordinate[1]) / (1000 * coordinates[0][1] - 1000 * coordinates[-1][1]))
    return abs(length * (1000 * cross[0] - 1000 * coordinate[0]) / (1000 * coordinates[0][0] - 1000 * coordinates[-1][0]))

# 给出路段，端点坐标，距离，返回坐标
# TODO 没考虑把road2跑完跑到road3的情况
def getCoordinate(road_id,cross,dis):
    coordinates,length = road_data[road_id]
    # 找哪个点里cross远
    c1 = coordinates[0]
    c2 = coordinates[-1]
    another_end = c2
    if abs(c1[0] - cross[0]) > abs(c2[0] - cross[0]) :
        another_end = c1
    x = cross[0] + dis / length * (another_end[0] - cross[0])
    y = cross[1] + dis / length * (another_end[1] - cross[1])
    return (x,y)


# 建立交点的坐标 
print("start get cross_data")   
cross_data = {}# (road_id,road_id) => cross
for i in range(n_road + 1):
    list = allowed_trans[i]
    for j in list :
        if j == i:
            continue
        coordinates_i,_ = road_data[i]
        coordinates_j,_ = road_data[j]
        cross = getCross(coordinates_i[0],coordinates_i[-1],coordinates_j[0],coordinates_j[-1])
        cross_data[(i,j)] = cross

# 开始读入jump_task.csv
print("start read jump_task.csv")
jump_task_data = []
start_speed = [-1 for i in range(len(traj_id2idx))] # 每条轨迹的起始速度
hour = [-1 for i in range(len(traj_id2idx))] # 每条轨迹的起始小时
holiday = [-1 for i in range(len(traj_id2idx))] # 每套轨迹是否发生于节假日
with open(cwd + '/../database/data/jump_task.csv','r') as file:
    csv_reader = csv.reader(file)
    head = 0
    for row in csv_reader:
        if head == 0:
            head = 1
            continue
        id = int(row[0])
        time = row[1]
        entity_id = int(row[2])
        traj_id = int(row[3])
        if (row[4] == ''):
            coordinate = None
        else :
            coordinate = parseCooridinate(row[4])
        if (row[5] == ''):
            current_distance = None
        else :
            current_distance = float(row[5])
        speeds = float(row[6])
        holidays = float(row[7])
        if traj_id in traj_id2idx.keys() :
            if start_speed[traj_id2idx[traj_id]] == -1:
                start_speed[traj_id2idx[traj_id]] = speeds
            if hour[traj_id2idx[traj_id]] == -1:
                hour[traj_id2idx[traj_id]] = getHour(time)
            if holiday[traj_id2idx[traj_id]] == -1:
                holiday[traj_id2idx[traj_id]] = holidays
        jump_task_data.append((id,time,entity_id,traj_id,coordinate,current_distance,speeds,holidays))
    file.close()

start_speed = np.array(start_speed)
hour = np.array(hour)
holiday = np.array(holiday)
serializable_road_ids = [subseq.tolist() if isinstance(subseq, np.ndarray) else subseq for subseq in road_ids]
# 保存为 JSON 文件
with open(cwd + '/../database/ETA/road_ids.json', 'w') as json_file:
    json.dump(serializable_road_ids, json_file)
np.save(cwd + '/../database/ETA/start_speed.npy',start_speed)
np.save(cwd + '/../database/ETA/hour.npy',hour)
np.save(cwd + '/../database/ETA/holiday.npy',holiday)
print(len(start_speed))

# 导入速度预测器
print('saved')

with open(cwd + '/../database/ETA/speed_for_jump.json', 'r') as f:
    mean_speed_per_road = json.load(f)


# 检测current_distance是否为None，是则开始计算
print("start calculate distance and coordinate")
with open('jump_task.csv',"w") as file :
    csv_writer = csv.writer(file)
    csv_writer.writerow(["id","time","entity_id","traj_id","coordinates","current_dis","speeds","holidays"])
    for i in range(len(jump_task_data)):
        id2,t2,entity_id,traj_id,coordinate2,current_distance2,speeds2,holidays2 = jump_task_data[i]
        if coordinate2 == None:
            id1,t1,_,_,coordinate1,current_distance1,speeds1,holidays1 = jump_task_data[i - 1]
            delta_t = time_difference(t1,t2)
            # 有可能出现fmm_jump匹配失败无法预测
            if (entity_id,traj_id) not in predict_data.keys() :
                current_distance2 = 'fmm匹配失败'
                coordinate2 = coordinate1
            else :
                road_id1,road_id2,road_id0 = predict_data[(entity_id,traj_id)]
                if road_id1 == road_id2 : # TODO 毒瘤情况
                    cross = cross_data[(road_id0,road_id1)]
                    d1 = getRemainDis(road_id1,cross,coordinate1)
                    delta_d = (speeds1 + speeds2) / 2  * delta_t
                    current_distance2 = current_distance1 + delta_d
                    coordinate2 = getCoordinate(road_id1,cross,d1+delta_d)
                else :
                    cross = cross_data[(road_id1,road_id2)]
                    index = traj_id2idx[traj_id]
                    lll = ll_data[(entity_id,traj_id)]
                    # 倒数第二条路段的速度v1
                    v1 = mean_speed_per_road[index][len(mean_speed_per_road[index]) - 2] * 60 / 1000
                    remainDis = getRemainDis(road_id1,cross,coordinate1)
                    t4road1 = remainDis / speeds1
                    if t4road1 < delta_t : # 说明开到road2上了
                        # 最后一条路段的速度v2
                        v2 = mean_speed_per_road[index][len(mean_speed_per_road[index]) - 1] * 60 / 1000
                        dis4road2 = speeds2 * (delta_t - t4road1)
                        coordinate2 = getCoordinate(road_id2,cross,dis4road2)
                        current_distance2 = current_distance1 + remainDis + dis4road2
                    else : # 说明没时间把road1开完
                        # 计算距离cross的距离
                        d = remainDis - speeds1 * delta_t
                        coordinate2 = getCoordinate(road_id1,cross,d)
                        current_distance2 = current_distance1 + speeds1 * delta_t

                
            
        # 将结果写入jump_task.csv
        # 这里没写回原文件,写在当前目录下的jump_task.csv中
        row = [id2,t2,entity_id,traj_id,"[{:.6f},{:.6f}]".format(coordinate2[0],coordinate2[1]),current_distance2,speeds2,int(holidays2)]
        csv_writer.writerow(row)
    file.close