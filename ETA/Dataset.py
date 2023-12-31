import os
import pickle
import numpy as np
import pandas as pd
import ast
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

class TrajDatasetNoGraph(Dataset):
    '''
    暂时不考虑GNN的Traj数据集，路段的编码暂时直接用原始编码.
    '''
    def __init__(self, path, minibatchsize) -> None:
        super().__init__()
        with open(path, "rb") as f:
            self.traj_data = pickle.load(f)  # a list of tuple
        self.size = len(self.traj_data)
        self.first_final_matched_points = np.array(list(map(lambda x:np.float32([x[1], x[2]]), self.traj_data)), dtype=np.float32)  # np.array
        self.traj_road_ids = list(map(lambda x: x[-1], self.traj_data))   # list of np.array
        self.traj_time = np.array(list(map(lambda x: np.float32(x[-2]), self.traj_data)), dtype=np.float32)
        self.start_speed = np.array(list(map(lambda x: np.float32(x[3]), self.traj_data)), dtype=np.float32)
        self.final_speed = np.array(list(map(lambda x: np.float32(x[4]), self.traj_data)), dtype=np.float32)
        self.batchsize = minibatchsize
        
    def __len__(self):
        return self.size
    
    def batch_generator(self, drop_last:bool = True, need_indices:bool = False):
        '''
        返回的都是np.ndarray
        [起点匹配点的位置，终点匹配点的位置] (ndarray, (3,)), \\
        [经过的路段的编号] (ndarray, dtype = int, 不定长) \\
        轨迹总时间 (float, in minutes) \\
        轨迹起始速度(float, in km/h) \\
        轨迹最终速度 (float, in km/h) \\
        除了road_ids之外，都是np.ndarray, road_ids是list of ndarray.
        '''
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.size)),
            self.batchsize,
            drop_last=drop_last
        )
        for indices in sampler:
            first_last_point = self.first_final_matched_points[indices]
            times = self.traj_time[indices]
            start_speed = self.start_speed[indices]
            final_speed = self.final_speed[indices]
            road_ids = [self.traj_road_ids[i] for i in indices]
            if need_indices:
                yield indices, first_last_point, road_ids, times, start_speed, final_speed
            else:
                yield first_last_point, road_ids, times, start_speed, final_speed

    def iter_traj_by_order(self, num_trajs_each_iter):
        i = 0
        while i < self.size:
            indices = range(i, i+num_trajs_each_iter if i+num_trajs_each_iter<=self.size else self.size)
            first_last_point = self.first_final_matched_points[indices]
            times = self.traj_time[indices]
            start_speed = self.start_speed[indices]
            final_speed = self.final_speed[indices]
            road_ids = [self.traj_road_ids[i] for i in indices]
            i += num_trajs_each_iter
            yield first_last_point, road_ids, times, start_speed, final_speed
    
    def __getitem__(self, index):
        '''
        返回的都是np.ndarray
        [起点匹配点的位置，终点匹配点的位置] (ndarray, (3,)), \\
        [经过的路段的编号] (ndarray, dtype = int, 不定长) \\
        轨迹总时间 (float, in minutes) \\
        轨迹起始速度(float, in km/h) \\
        轨迹最终速度 (float, in km/h)
        '''
        traj = self.traj_data[index]
        d = np.array([traj[1], traj[2]], dtype=np.float32)
        p = np.array(traj[-1], dtype=np.int32)
        t = traj[-2]
        ssp = traj[3]
        fsp = traj[4]
        return d, p, t, ssp, fsp
    
class RoadFeatures:
    '''
    暂时不考虑图神经网络。根据路段的id加载道路的feature
    训练完毕之后，可以提前将图中路段编码好，在推理的时候可以直接用它.
    '''
    def __init__(self, feat_path, attr_path) -> None:
        with open(feat_path, "rb") as f:
            self.feature = np.float32(pickle.load(f))  # ndarray: (num_road, num_features)
        self.n_features = self.feature.shape[1]
        self.road_attr = pd.read_csv(attr_path)
        self.road_length = np.float32(self.road_attr['length'].values)   # ndarray数组. 一维
        self.road_coords = list(map(lambda x: ast.literal_eval(x), self.road_attr['coordinates']))  # 三维list, 每条路段时 [二维点，二维点...]
        self.road_start_coord = np.array(list(map(lambda x: x[0], self.road_coords)), dtype=np.float32)  # 二维list, 每个路段的起点坐标.
        self.road_end_coord = np.array(list(map(lambda x: x[-1], self.road_coords)), dtype=np.float32)   # 二维list, 每个路段的终点坐标.
        
    
    def getFeatures(self, idx):
        '''
        idx可以是一个数组.
        '''
        return self.feature[idx]
    
    def getRoadLength(self, idx):
        '''
        路段长度单位是m(应该.), 结果是一维np.ndarray数组
        '''
        return self.road_length[idx]
    
    def getRoadOrigin(self, idx):
        '''
        路段的起点坐标，结果是ndarray, 如果idx是数组，结果也是二维的；如果只是一个数，则结果是一维的.
        '''
        return self.road_start_coord[idx]
    
    def getRoadTarget(self, idx):
        '''
        路段的终点坐标，结果是ndarray, 如果idx是数组，结果也是二维的；如果只是一个数，则结果是一维的.
        '''
        return self.road_end_coord[idx]