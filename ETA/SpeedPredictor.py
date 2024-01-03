import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_, constant_, kaiming_normal_
from torch.utils.tensorboard import SummaryWriter
import pickle
import json
import os

from Dataset import RoadFeatures
from Utils import *
from GRU import GRUmodelBoosting

class SpeedPredictor:
    def __init__(self) -> None:
        '''
        参数已经预先写好.不用传什么东西.
        '''
        self.base_dir = "ETA/newBoosting/"
        device = torch.device("cuda") if torch.device.is_available() else torch.device("cpu")
        self.rawroadfeat = RoadFeatures("ETA/road_features_with_lengths.pkl", "database/data/road.csv")
        self.model = GRUmodelBoosting(self.rawroadfeat.n_features, device, 9, params_path=self.base_dir)
    

    def predict_speed(self, road_ids, start_speed, hour, holiday):
        '''
        参数与类型解释： \\
        road_ids: 二维列表。有batchsize个元素，每个都是一个所经过的路段编号的列表, 比如，轨迹a依次经过1,2,3，轨迹b依次经过7,8, 则两者合起来为[[1,2,3],[7,8]] \\
        start_end_points: np.ndarray, shape为(batchsize, 2), start_end_points[:,0]为起点坐标, start_end_points[:,1]为终点坐标 \\
        start_speed: np.ndarray, shape为(batchsize,)，每个轨迹的起始速度.  \\
        hour: np.ndarray, shape为(batchsize, )，每个轨迹时间中的小时. 如开始时间为8:30的轨迹这个值为8. \\
        holiday: np.ndarray, shape为(batchsize, ), 轨迹是否发生于假日, 为0-1(float). \\
        
        返回值： \\
        mean_speed_per_road: np.ndarray, shape为(length, batchsize). 其中，轨迹的长度定义为轨迹所经过的不同路段的数量，length为这一个batch中最长的路段的长度.
        长度不足length的轨迹，不足的部分**不一定是0**。比如，mean_speed_per_road[2,3]为batch中的第4条轨迹在其所经过的第3个路段上的平均速度。 \\
        虽然预测结果有所改进，但仍是不准的.
        '''
        return self.model.predict_speed(road_ids, start_speed, self.rawroadfeat, hour, holiday).detach().cpu().numpy()