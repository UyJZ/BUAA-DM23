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
from GRU import GRUmodel
from Utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trajSet = TrajDatasetNoGraph("ETA/traj_data_train.pkl", 8)
rawroadfeat = RoadFeatures("ETA/road_features_with_lengths.pkl", "database/data/road.csv")

is_training = False
hidden_size = 128
lr = 5e-5
max_grad_norm = 1
num_epoch = 500
log_dir = "ETA/GRUonly_result/losslog"
param_path = "ETA/GRUonly_result/rawfeature_reg_gru.pt"
learnable_init_hidden = False
torch.manual_seed(5017)

gru = GRUmodel(rawroadfeat.n_features, hidden_size, learnable_init_hidden_code=learnable_init_hidden).to(device)
if not is_training:
    gru.load_state_dict(torch.load(param_path))
    gru.eval()
else:
    optimizer = torch.optim.Adam(gru.parameters(), lr, weight_decay=0.001)


if is_training:
    summary = SummaryWriter(log_dir)

    global_steps = 0
    epoches_per_save = 20
    epoches_before_save = 0
    steps_per_print = 200
    steps_before_print = 0

    for e in range(num_epoch):
        epoch_start_time = time.time()
        for traj_info, road_ids, duration, start_speed, final_speed in trajSet.batch_generator():
            feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
            feat_sequence = feat_sequence.to(device)
            start_speed = torch.FloatTensor(start_speed).to(device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
            final_speed = torch.FloatTensor(final_speed).to(device) * 1000 / 60
            
            if learnable_init_hidden:
                h0 = start_speed.unsqueeze(0).unsqueeze(2)
            else:
                h0 = get_positional_encoding(start_speed, hidden_size)   # (batchsize, hiddensize)
                h0[:,0] = start_speed
                h0.unsqueeze_(0)
            #h0 = torch.concat((start_speed.unsqueeze(1), h0), dim=1).unsqueeze(0).to(device)  # 变成(1, batchsize, hidden_size)
            mean_speed_per_road, predicted_final_speed = gru(feat_sequence, h0)  # mean_speed: (L,T,1), final_speed: (T,1)
            #print(mean_speed_per_road[:,0,0])

            # 有了路段平均速度，预测的最终速度，然后求时间、算loss.
            road_lengths_sequence = prepare_road_lengths(rawroadfeat, road_ids, traj_info[0], traj_info[1]).to(device)  # (L,T,1)

            # t = road_lengths_sequence / (mean_speed_per_road + 1e-6)
            # print(t[:,0,0])
            times = torch.sum(road_lengths_sequence / (mean_speed_per_road + 1e-6), dim=0).squeeze()    # 这样得到的应该是 (T,)
            #print(times.data)
            if steps_before_print == steps_per_print:
                print(times.data)
                steps_before_print = 0
            else:
                steps_before_print += 1
            
            label_times = torch.FloatTensor(duration).to(device)
            #print(times.dtype, label_times.dtype, predicted_final_speed.dtype, final_speed.dtype)
            loss_times = F.mse_loss(times, label_times)
            loss_finalspeed = F.mse_loss(predicted_final_speed.squeeze(), final_speed)
            # 记录loss_times, loss_finalspeed...
            summary.add_scalar("Loss of Time", loss_times, global_steps)
            summary.add_scalar("Loss of Final Speed", loss_finalspeed, global_steps)

            L = loss_times + loss_finalspeed + 0.01 * torch.sum(times)

            optimizer.zero_grad()
            L.backward()
            global_steps += 1

            nn.utils.clip_grad_norm_(gru.parameters(), max_grad_norm)
            optimizer.step()
        
        # print("last batch of epoch:")
        # print(times.data)
        # print(label_times.data)
        if epoches_before_save == epoches_per_save:
            torch.save(gru.state_dict(), param_path)
            epoches_before_save = 0
            print("Parameters saved.".format(epoches_per_save))
        else:
            epoches_before_save += 1

    torch.save(gru.state_dict(), param_path)


else:
    torch.no_grad()
    ret_table = np.zeros((len(trajSet), 2), np.float32)  # label_time, predicted_time.
    l = 0
    for traj_info, road_ids, duration, start_speed, final_speed in trajSet.batch_generator(drop_last=False):
        feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
        feat_sequence = feat_sequence.to(device)
        start_speed = torch.FloatTensor(start_speed).to(device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
        final_speed = torch.FloatTensor(final_speed).to(device) * 1000 / 60
        
        if learnable_init_hidden:
            h0 = start_speed.unsqueeze(0).unsqueeze(2)
        else:
            h0 = get_positional_encoding(start_speed, hidden_size)   # (batchsize, hiddensize)
            h0[:,0] = start_speed
            h0.unsqueeze_(0)
        mean_speed_per_road, predicted_final_speed = gru(feat_sequence, h0)  # mean_speed: (L,T,1), final_speed: (T,1)

        # 有了路段平均速度，预测的最终速度，然后求时间
        road_lengths_sequence = prepare_road_lengths(rawroadfeat, road_ids, traj_info[0], traj_info[1]).to(device)  # (L,T,1)

        times = torch.sum(road_lengths_sequence / (mean_speed_per_road + 1e-6), dim=0).squeeze().detach().cpu().numpy()    # 这样得到的应该是 (T,)
        label_times = np.array(duration, dtype=np.float32)

        ret_table[l:l+times.shape[0], 0] = label_times
        ret_table[l:l+times.shape[0], 1] = times
        l += times.shape[0]
    
    # 平均绝对值误差.
    loss = np.abs(ret_table[:,0] - ret_table[:,1])
    mean_loss = np.mean(loss)
    std = np.std(loss)
    #print(ret_table)
    print(mean_loss)
    print(std)

    # 保存ret_table.
    df = pd.DataFrame(ret_table,columns=["label time", "predicted time"])
    df.to_csv("ETA/GRUonly_result/train.csv", sep=',', index=False)