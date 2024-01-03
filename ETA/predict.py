import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from Dataset import TrajDatasetNoGraph, RoadFeatures
from GRU import GRUmodelBoosting
from Utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trajSet = TrajDatasetNoGraph("ETA/traj_data_test.pkl", 8, hour_holiday="ETA/traj_hour_holiday_test.pkl")
rawroadfeat = RoadFeatures("ETA/road_features_with_lengths.pkl", "database/data/road.csv")

base_dir = "ETA/newBoosting/"

model = GRUmodelBoosting(rawroadfeat.n_features, device, 9, params_path=base_dir).to(device)

ret = (predict_all_trajs(trajSet, model, rawroadfeat)).detach().cpu().numpy()
df = pd.DataFrame(ret, columns=["real time", "predict"])
df.to_csv(os.path.join(base_dir, "ret_test.csv"), index=False)