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
import signal

from Dataset import TrajDatasetNoGraph, RoadFeatures
from GRU import GRUmodel, SimpleGRU, GRUBoosting, GRUBagging
from Utils import *

stopped = False

is_training = False
#nbags = 8
nbags = 6   # 筛选出来6个

trajSet = TrajDatasetNoGraph("ETA/traj_data_train.pkl", 8, nbags) if is_training else TrajDatasetNoGraph("ETA/traj_data_train.pkl", 8)
rawroadfeat = RoadFeatures("ETA/road_features_with_lengths.pkl", "database/data/road.csv")
base_bagging_dir = "ETA/GRUBagging"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GRUs = GRUBagging(rawroadfeat.n_features, device, nbags, is_training=is_training, params_path=base_bagging_dir)


if is_training:
    lr = 5e-5
    max_grad_norm = 1
    num_epoch = 20
    log_dir = os.path.join(base_bagging_dir, "losslog")
    torch.manual_seed(5017)
    with open("ETA/traj_restime_train.pkl", "rb") as f:
        label_restimes = torch.from_numpy(pickle.load(f)).to(device)
    summary = SummaryWriter(log_dir)

    for K in range(nbags):
    #for K in [0,4,7]:   # 发现这个不行.
        optimizer = torch.optim.Adam(GRUs.GRUs[K].parameters(), lr, weight_decay=0.01)
        global_steps = 0
        for e in range(num_epoch):
            for indices, traj_info, road_ids, duration, start_speed, final_speed in trajSet.batch_generator(need_indices=True, bootstrap_id=K):
                predicted = GRUs.predict(road_ids, start_speed, rawroadfeat, K)
                label = label_restimes[indices]

                L = F.mse_loss(predicted, label)
                summary.add_scalar("Loss of GRU {}".format(K), L, global_steps)
                global_steps += 1

                optimizer.zero_grad()
                L.backward()
                nn.utils.clip_grad_norm_(GRUs.GRUs[K].parameters(), max_grad_norm)
                optimizer.step()
            #print("trained an epoch")
        torch.save(GRUs.GRUs[K].state_dict(), os.path.join(base_bagging_dir, "param{}.pt".format(K)))
        print("GRU {}'s parameters saved.".format(K))

else:
    fundGRU = GRUmodel(rawroadfeat.n_features, 128, device).eval().to(device)
    fundGRU.load_state_dict(torch.load("ETA/GRUonly_result/rawfeature_reg_gru.pt"))
    ret1 = predict_all_trajs(trajSet, fundGRU, rawroadfeat).detach().cpu().numpy()

    ret2 = predict_all_trajs(trajSet, GRUs, rawroadfeat).detach().cpu().numpy()
    ret1[:,1] = np.clip(ret1[:,1] + ret2[:,1], 2, 1000)
    
    df = pd.DataFrame(ret1, columns=["label time", "predicted time"])
    df.to_csv(os.path.join(base_bagging_dir, "ret_train.csv"))