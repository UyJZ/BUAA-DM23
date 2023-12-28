import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time

from ETA.Dataset import TrajDatasetNoGraph, RawRoadFeatures
from ETA.GRU import GRUmodel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trajSet = TrajDatasetNoGraph("ETA/traj_data_train.pkl")
trajloader = DataLoader(trajSet, batch_size=8, shuffle=True)
rawroadfeat = RawRoadFeatures("road_features.pkl")

gru = GRUmodel(rawroadfeat.n_features, 64).to(device)

num_epoch = 100

for e in range(num_epoch):
    start_time = time.time()
    for traj, label in trajloader:
        pass