import os
import numpy as np
import pandas as pd
import ast

df_road = pd.read_csv("database/data/road.csv")
df_rel = pd.read_csv("database/data/rel.csv")

num_road = df_road['id'].count()
num_rel = df_rel['id'].count()

# 形成邻接表
graph_to = [[] for _ in range(num_road)]    # 这里是出边
graph_from = [[] for _ in range(num_road)]  # 这里是入边
for idx, rel in df_rel.iterrows():
    origin = int(rel[1])
    target = int(rel[2])
    graph_to[origin].append(target)
    graph_from[target].append(target)


# 定义路段相似度.
def shapeSim(row1, row2):
    '''
    这个函数左右不对称，所以应该左右两边各算一次，取平均值或者最大值.
    '''
    coord1 = np.array(ast.literal_eval(row1[1]) if type(row1[1])==str else row1[1])
    coord1 = (coord1 - coord1[0,:]) * 1000
    coord2 = np.array(ast.literal_eval(row2[1]) if type(row2[1])==str else row2[1])
    coord2 = (coord2 - coord2[0,:]) * 1000
    # 总长度归一化. ——暂时先不归一化了..
    # 插值缺少的点.
    num_points1 = coord1.shape[0]
    num_points2 = coord2.shape[0]
    diff = num_points1 - num_points2
    skip_shape = False    # shape 相似度直接为0
    if diff > 0:
        if diff > num_points2 - 1:
            skip_shape = True
        else:
            coord2 = addPoints(coord2, diff)
    elif diff<0:
        diff = -diff
        if diff > num_points1 - 1:
            skip_shape = True
        else:
            coord1 = addPoints(coord1, diff)
    if not skip_shape:
        # 变成齐次坐标.
        coord1 = coord1[1:,:] - coord1[:-1,:]
        coord2 = coord2[1:,:] - coord2[:-1,:]
        # print(coord1)
        # print(coord2)
        coord1 = np.concatenate((coord1, np.ones((coord1.shape[0], 1))), axis=1)
        coord2 = np.concatenate((coord2, np.ones((coord2.shape[0], 1))), axis=1)
        delta = 0
        # 因为是行向量，所以应该是coord2 = coord1 * A
        if coord1.shape[0] > coord1.shape[1]:     # 可以求伪逆.
            coord1_pinv = np.linalg.pinv(coord1)
            #print(coord1_pinv.shape, coord2.shape, coord1.shape)
            A = np.matmul(coord1_pinv, coord2)
            #print(A)
            # test = np.matmul(coord1, A)
            # test = test[:,:-1] / (test[:,-1][:,np.newaxis] + 1e-6)
            # delta = np.sum(np.abs(coord2[:,:-1] - test))
            delta += np.abs(np.sum(A**2) - 3)
            coord2_pinv = np.linalg.pinv(coord2)
            A = np.matmul(coord2_pinv, coord1)
            delta += np.abs(np.sum(A**2) - 3)
        else:     # 不可以求伪逆，用最小二乘法.
            A = np.linalg.lstsq(coord1, coord2, rcond=None)[0]
            #print(A)
            delta += np.abs(np.sum(A**2) - coord1.shape[0])
            A = np.linalg.lstsq(coord2, coord1, rcond=None)[0]
            delta += np.abs(np.sum(A**2) - coord1.shape[0])
        # 用Sigmoid函数，加点变换.
        # 1/(1+e^{10x-5})
        #shape_sim = 1/(1+np.exp(10*delta - 5))
        delta /= 2
        shape_sim = 1/(1+np.exp(delta - 5)) if delta < 20 else 0
    else:
        shape_sim = 0

    return shape_sim

def addPoints(pointArr:np.ndarray, addnum:int):
    '''
    约束addnum < 点个数-1（线段个数），也就是最多插值一次.
    '''
    newpoints = (pointArr[1:,:] + pointArr[:-1,:]) / 2
    ret = []
    i = 0
    while i<addnum:
        ret.append(pointArr[i,:].flatten())
        ret.append(newpoints[i,:].flatten())
        i += 1
    while i< pointArr.shape[0]:
        ret.append(pointArr[i,:].flatten())
        i += 1
    return np.array(ret)


def topoSim(row1, row2, highway_coef = 0.5, topo_coef = 0.5):
    highwayTypeNum = 14
    id1 = int(row1[0])
    id2 = int(row2[0])
    highway1 = int(row1[2])
    highway2 = int(row2[2])
    highwaySim = float(highway1 == highway2)

    # 考虑的话是在太慢了，图中前后连接的话实在太慢，不要一个一个算.
    # topoSim = 0
    # for graph in [graph_from, graph_to]:
    #     total = len(graph[id1]) + len(graph[id2])
    #     if total == 0:
    #         #topoSim += 1
    #         continue
    #     highway_cnt1 = [0 for _ in range(highwayTypeNum)]
    #     highway_cnt2 = [0 for _ in range(highwayTypeNum)]
    #     for i in graph_from[id1]:
    #         from_highway = int(df_road.iloc[id1][2])
    #         highway_cnt1[from_highway] += 1
    #     for i in graph_from[id2]:
    #         from_highway = int(df_road.iloc[id2][2])
    #         highway_cnt2[from_highway] += 1
    #     cnt1 = np.array(highway_cnt1)
    #     cnt2 = np.array(highway_cnt2)
    #     s = np.sum(np.where(cnt1<cnt2, cnt1, cnt2))
    #     topoSim += s / (total)
    # topoSim /= 2
    return highwaySim

def roadSim(row1, row2):
    #shape_sim = min(shapeSim(row1, row2), shapeSim(row2, row1))
    shape_sim = shapeSim(row1, row2)
    topo_sim = topoSim(row1, row2)
    return (shape_sim + topo_sim) / 2


# 因为求平均比较困难，所以用DBScan.
from sklearn.cluster import DBSCAN
import pickle
from tqdm import tqdm
import multiprocessing
from numba import jit
import time


metric = np.zeros((num_road, num_road))

@jit
def computeDistance():
    ori = time.time()
    for i in range(num_road):
        thisroad = df_road.iloc[i]
        for j in range(num_road):
            if i>j:
                metric[i,j] = metric[j,i]
            elif i<j:
                metric[i,j] = roadSim(thisroad, df_road.iloc[j])
        t = time.time()
        print("\riter {}. taked {}s".format(i, t-ori), end="")
        ori = t

computeDistance()

@jit
def computeDistance_0():
    unit = num_road // 3
    m = np.zeros((unit, num_road))
    ori = time.time()
    for i in range(0, unit):
        for j in range(num_road):
            # if i<j:
            #     metric[i,j] = metric[j,i]
            # elif i>j:
            #     metric[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
            m[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
        t = time.time()
        print("\riter: {}, taked {}s".format(i, t - ori))
        ori = t
    metric[:unit,:] = m

@jit
def computeDistance_1():
    unit = num_road // 3
    m = np.zeros((unit, num_road))
    for i in tqdm(range(unit, 2*unit)):
        for j in range(num_road):
            # if i<j:
            #     metric[i,j] = metric[j,i]
            # elif i>j:
            #     metric[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
            m[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
    metric[unit:2*unit, :] = m

@jit
def computeDistance_2():
    unit = num_road // 3
    m = np.zeros((num_road - 2*unit, num_road))
    for i in tqdm(range(2*unit, num_road)):
        for j in range(num_road):
            # if i<j:
            #     metric[i,j] = metric[j,i]
            # elif i>j:
            #     metric[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
            m[i,j] = roadSim(df_road.iloc[i], df_road.iloc[j])
    metric[2*unit:num_road, :] = m

# multiprocessing.set_start_method("spawn")
# p1 = multiprocessing.Process(target=computeDistance_0)
# p2 = multiprocessing.Process(target=computeDistance_1)
# p3 = multiprocessing.Process(target=computeDistance_2)
# for p in [p1,p2,p3]:
#     p.start()
# p1.join()
# p2.join()
# p3.join()


with open("DisMat.pkl", "wb") as f:
    pickle.dump(metric, f)

dbscan = DBSCAN(0.4, min_samples=500, metric="precomputed")
ret = dbscan.fit_predict(1-metric)
with open("ClusterRet.pkl", "wb") as f:
    pickle.dump(ret, f)