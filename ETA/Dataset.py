import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class TrajDatasetNoGraph(Dataset):
    '''
    暂时不考虑GNN的Traj数据集，路段的编码暂时直接用原始编码.
    '''
    def __init__(self, path) -> None:
        super().__init__()
        with open(path, "rb") as f:
            self.traj_data = pickle.load(f)
        
    def __len__(self):
        return len(self.traj_data)
    
    def __getitem__(self, index):
        '''
        数据：[起点匹配点的位置，终点匹配点的位置，初始速度，经过的路段编号(np一维数组, int.)], label: [轨迹总时间，轨迹最终速度], 都是list.
        '''
        traj = self.traj_data[index]
        d = [traj[1], traj[2], traj[3], np.array(traj[-1])]
        l = [traj[-2], traj[4]]
        return d, l
    
class RawRoadFeatures:
    '''
    暂时不考虑图神经网络。根据路段的id加载道路的feature
    '''
    def __init__(self, path) -> None:
        with open(path, "rb") as f:
            self.raw_feature = pickle.load(f)  # ndarray: (num_road, num_features)
        self.n_features = self.raw_feature.shape[1]
    
    def getFeatures(self, idx):
        '''
        idx可以是一个数组.
        '''
        return self.raw_feature[idx]