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
from GRU import GRUmodel, SimpleGRU, GRUBoosting
from Utils import *

stopped = False

is_training = False

trajSet = TrajDatasetNoGraph("ETA/traj_data_test.pkl", 8)
rawroadfeat = RoadFeatures("ETA/road_features_with_lengths.pkl", "database/data/road.csv")
base_boosting_dir = "ETA/GRUBoosting"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if is_training:
    gruboosting = GRUBoosting(base_boosting_dir, is_training, rawroadfeat.n_features, 1, device=device, training_set_len=len(trajSet))
else:
    fundGRU = GRUmodel(rawroadfeat.n_features, 128, device).eval().to(device)
    fundGRU.load_state_dict(torch.load("ETA/GRUonly_result/rawfeature_reg_gru.pt"))
    gruboosting = GRUBoosting(base_boosting_dir, is_training, rawroadfeat.n_features, 1, device=device, training_set_len=len(trajSet), fundGRU=fundGRU)

if is_training:

    def handle_ctrlc(signum, frame):
        stopped = True
        return
    signal.signal(signal.SIGINT, handle_ctrlc)

    lr = 5e-5
    max_grad_norm = 1
    num_epoch = 500
    log_dir = "ETA/GRUBoosting/losslog{}".format(gruboosting.training_layer)
    torch.manual_seed(5017)
    with open("ETA/traj_restime_train.pkl", "rb") as f:
        label_restimes = torch.from_numpy(pickle.load(f)).to(device)

    optimizer = torch.optim.Adam(gruboosting.getTrainingParams(), lr, weight_decay=0.01)
    
    summary = SummaryWriter(log_dir)
    global_steps = 0

    epoches_per_save = 5
    epoches_before_save = 0
    n_epoches = 100
    for e in range(n_epoches):
        if stopped:
            break
        for indices, traj_info, road_ids, duration, start_speed, final_speed in trajSet.batch_generator(need_indices=True):
            if stopped:
                break
            dist = gruboosting.getDataDistribution(indices)  # distribution.  注意是乘以了len(trajSet)的
            predicted = gruboosting.predict(road_ids, traj_info, start_speed, rawroadfeat) * dist
            label = label_restimes[indices] * dist
            # 加权平均平方误差.
            #L = torch.sum((label - predicted) ** 2 * dist) / 2
            L = F.mse_loss(predicted, label)
            summary.add_scalar("loss of weighted restime", L, global_steps)
            global_steps += 1
            if global_steps % 500 == 0:
                print("predicted: {}\nloss: {}".format(predicted, L))

            optimizer.zero_grad()
            L.backward()
            nn.utils.clip_grad_norm_(gruboosting.getTrainingParams(), max_grad_norm)
            optimizer.step()
        
        epoches_before_save += 1
        if epoches_before_save == epoches_per_save:
            ret = predict_all_trajs(trajSet, gruboosting, rawroadfeat, label_restimes)
            gruboosting.stop_training(ret[:,0], ret[:,1])
            print("save parameters")
            epoches_before_save = 0
    
    ret = predict_all_trajs(trajSet, gruboosting, rawroadfeat, label_restimes)
    gruboosting.stop_training(ret[:,0], ret[:,1])
    print("Training ends. Save parameters and update distributions")

else:
    ret = predict_all_trajs(trajSet, gruboosting, rawroadfeat).detach().cpu().numpy()
    df = pd.DataFrame(ret, columns=['label','predicted'])
    df.to_csv("ETA/GRUBoosting/ret_test.csv")