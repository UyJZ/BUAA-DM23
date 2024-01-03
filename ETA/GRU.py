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

class GRUmodel(nn.Module):
    '''
    当前的设想：用一个单层的GRU完成序列中每个路段上的平均速度的预测. \\
    轨迹构成一个序列，每个路段的输入是当前路段的长度连上它的编码，然后初始隐藏状态时其初始速度v0, 并且扩充成和隐藏层相同维度的张量.. \\
    用一个共享的神经网络(暂时用一个单层的神经网络.)将序列中每个路段的输出结果(是一个和隐藏状态同纬度的东西)变换为这个路段上的平均速度.  \\
    最后的隐藏状态也通过这个神经网络，与轨迹的末尾速度作一个正则化..? \\
    暂时不考虑路口的延迟.
    '''
    def __init__(self, input_dim, hidden_dim, device, dropout=0.0, learnable_init_hidden_code=False, n_hidden_hour=60, n_hidden_holiday=8, n_hidden_speeds=60) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout = dropout).to(device)
        self.linear = nn.Linear(hidden_dim, hidden_dim//2).to(device)
        self.linear1= nn.Linear(hidden_dim//2, 1).to(device)
        self.device = device

        self.hidden_size = hidden_dim

        self.n_hidden_hour = n_hidden_hour
        self.n_hidden_holiday = n_hidden_holiday
        self.n_hidden_speeds = n_hidden_speeds

        # orthogonal_初始化
        orthogonal_(self.gru.weight_ih_l0, gain=2)
        orthogonal_(self.gru.weight_hh_l0, gain=2)
        constant_(self.gru.bias_ih_l0, 0.0)
        constant_(self.gru.bias_hh_l0, 0.0)
        orthogonal_(self.linear.weight, gain=2)
        #kaiming_normal_(self.linear.weight)
        constant_(self.linear.bias, 0.0)
        orthogonal_(self.linear1.weight, gain=2)
        #kaiming_normal_(self.linear1.weight)
        constant_(self.linear1.bias, 0.0)

        self.learn_h0 = learnable_init_hidden_code
        if learnable_init_hidden_code:
            self.h0_linear0 = nn.Linear(1, hidden_dim//2).to(device)
            self.h0_linear1 = nn.Linear(hidden_dim//2, hidden_dim).to(device)
            orthogonal_(self.h0_linear0.weight, gain=2)
            #kaiming_normal_(self.h0_linear0.weight)
            constant_(self.h0_linear0.bias, 0.0)
            orthogonal_(self.h0_linear1.weight, gain=2)
            #kaiming_normal_(self.h0_linear1.weight)
            constant_(self.h0_linear1.bias, 0.0)

    def forward(self, x, init_hidden):
        '''
        x: (sequence_length, batchsize, features), init_hidden: (1, batchsize, hidden_size); \\
        如果是learnable_init_hidden，则输入的是 (1, batchsize, 1)的tensor，表示初始速度；否则就是初始的hidden_code; \\
        x应该是一个PackedSequence, 具体是先用0把短的序列填满，然后调用 torch.nn.utils.rnn.pack_padded_sequence(). \\
        经过rnn后会用 torch.nn.utils.rnn.pad_packed_sequence 变回去(有pad过的).  \\
        返回值 mean_speed: (sequence_length, batchsize, 1) 表示每个路段的平均速度，还有一个 final_speed: (batchsize, 1) 表示最终速度(从最后一个隐藏层来的.)
        '''
        if self.learn_h0:
            init_hidden = self.h0_linear1(F.relu(self.h0_linear0(init_hidden)))
        outputs, hn = self.gru(x, init_hidden)  # hn: (1, batchsize, hidden_size)
        outputs_tensor, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # outputs_tensor: (sequence_length, batchsize, hidden_size) 后面那个_是忽略了batch中每个的长度.
        speeds = self.linear1(F.relu(self.linear(torch.cat([outputs_tensor, hn], dim=0)))).exp()  # 非负约束
        return speeds[:-1,:,:], speeds[-1,:,:]
    

    def predict(self, road_ids, start_end_points, start_speed, rawroadfeat:RoadFeatures, hour=None, holiday=None):
        '''
        road_ids: list. 其他start_end_points, start_speed都是numpy.ndarray.  \\
        返回：预测出来的时间，(batchsize,), tensor.
        '''
        with torch.no_grad():
            feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
            feat_sequence = feat_sequence.to(self.device)
            start_speed = torch.FloatTensor(start_speed).to(self.device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
            #final_speed = torch.FloatTensor(final_speed).to(device) * 1000 / 60
            
            if self.learn_h0:
                h0 = start_speed.unsqueeze(0).unsqueeze(2)
            elif hour is not None and holiday is not None:
                hour = torch.FloatTensor(hour).to(self.device)
                holiday = torch.FloatTensor(holiday).to(self.device)
                h0 = prepare_input(hour, holiday, start_speed, self.n_hidden_hour, self.n_hidden_holiday, self.n_hidden_speeds)
            else:
                h0 = get_positional_encoding(start_speed, self.hidden_size)   # (batchsize, hiddensize)
                h0[:,0] = start_speed
                h0.unsqueeze_(0)
            mean_speed_per_road, _ = self.forward(feat_sequence, h0)  # mean_speed: (L,T,1), final_speed: (T,1)

            # 有了路段平均速度，预测的最终速度，然后求时间
            road_lengths_sequence = prepare_road_lengths(rawroadfeat, road_ids, start_end_points[0], start_end_points[1]).to(self.device)  # (L,T,1)

            times = torch.sum(road_lengths_sequence / (mean_speed_per_road + 1e-6), dim=0).squeeze()    # 这样得到的应该是 (T,)
        return times
    
    def predict_speed(self, road_ids, start_speed, rawroadfeat:RoadFeatures, hour=None, holiday=None):
        with torch.no_grad():
            feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
            feat_sequence = feat_sequence.to(self.device)
            start_speed = torch.FloatTensor(start_speed).to(self.device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
            #final_speed = torch.FloatTensor(final_speed).to(device) * 1000 / 60
            
            if self.learn_h0:
                h0 = start_speed.unsqueeze(0).unsqueeze(2)
            elif hour is not None and holiday is not None:
                hour = torch.FloatTensor(hour).to(self.device)
                holiday = torch.FloatTensor(holiday).to(self.device)
                h0 = prepare_input(hour, holiday, start_speed, self.n_hidden_hour, self.n_hidden_holiday, self.n_hidden_speeds)
            else:
                h0 = get_positional_encoding(start_speed, self.hidden_size)   # (batchsize, hiddensize)
                h0[:,0] = start_speed
                h0.unsqueeze_(0)
            mean_speed_per_road, _ = self.forward(feat_sequence, h0)  # mean_speed: (L,T,1), final_speed: (T,1)

        return mean_speed_per_road.squeeze()  # (L,T)


class GRUmodelBagging(nn.Module):
    def __init__(self, input_dim, device, n_bagging, dropout=0.0, is_training = False, params_path:str = None) -> None:
        super().__init__()
        self.GRUs = [GRUmodel(input_dim, 128, device).to(device) for _ in range(n_bagging)]
        self.device = device
        self.nbags = n_bagging
        self.is_training = is_training
        if not is_training:
            pt_filename = []
            for f in os.listdir(params_path):
                if f.endswith('.pt'):
                    pt_filename.append(f)
            assert len(pt_filename) == n_bagging
            for i in range(n_bagging):
                self.GRUs[i].load_state_dict(torch.load(os.path.join(params_path, pt_filename[i])))
                self.GRUs[i].eval()

    def predict(self, road_ids, start_end_points, start_speed, rawroadfeat:RoadFeatures, id:int = None, hour = None, holiday = None):
        if self.is_training:
            return self.GRUs[id].predict(road_ids, start_end_points, start_speed, rawroadfeat, hour, holiday)
        else:
            t = [gru.predict(road_ids, start_end_points, start_speed, rawroadfeat, hour, holiday).unsqueeze(0) for gru in self.GRUs]
            t = torch.concat(t, 0)
            return torch.sum(t, dim=0) / self.nbags
        

class GRUmodelBoosting(nn.Module):
    def __init__(self, input_dim, device, n_boosting, dropout = 0.0, params_path:str="ETA/newBoosting/") -> None:
        super().__init__()
        self.GRUs = [GRUmodel(input_dim, 128, device, dropout=dropout).to(device) for _ in range(n_boosting)]
        self.device = device
        self.nboost = n_boosting
        
        pts = []
        for fname in os.listdir(params_path):
            if fname.endswith(".pt"):
                pts.append(fname)
        assert len(pts) == self.nboost
        pts.sort()
        for i in range(n_boosting):
            self.GRUs[i].load_state_dict(torch.load(os.path.join(params_path, pts[i])))
            self.GRUs[i].eval()
        with open(os.path.join(params_path, "alpha.pkl"), "rb") as f:
            self.alpha = torch.from_numpy(pickle.load(f)).to(device).unsqueeze(1)

    def predict(self, road_ids, start_end_points, start_speed, rawroadfeat:RoadFeatures, hour=None, holiday=None):
        with torch.no_grad():
            rets = [gru.predict(road_ids, start_end_points, start_speed, rawroadfeat, hour, holiday).unsqueeze(0) for gru in self.GRUs]  #[(1, batchsize,)]
            rets = torch.cat(rets, dim=0) * self.alpha #(n_boosting, batchsize)
            rets = torch.sum(rets, dim=0)
        return rets
    
    def predict_speed(self, road_ids, start_speed, rawroadfeat:RoadFeatures, hour=None, holiday=None):
        with torch.no_grad():
            speeds = [gru.predict_speed(road_ids, start_speed, rawroadfeat, hour, holiday).unsqueeze(0) for gru in self.GRUs]
            speeds = torch.cat(speeds, dim=0) * self.alpha
            speeds = torch.sum(speeds, dim=0)
        return speeds  # (L,T)


class SimpleGRU(nn.Module):
    '''
    hidden code只有1维，就是速度. 初始状态1维，就是初始速度.
    '''
    def __init__(self, input_dim, device, dropout = 0) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, 1, dropout=dropout)   # 不知道为什么加了dropout=0之后会报错需要在[0,1]范围....
        orthogonal_(self.gru.weight_ih_l0, gain=2)
        orthogonal_(self.gru.weight_hh_l0, gain=2)
        constant_(self.gru.bias_ih_l0, 0.0)
        constant_(self.gru.bias_hh_l0, 0.0)
        self.device = device
        self.gru.to(device)

    def forward(self, x, h0):
        '''
        x 是packedpadsequence, h0是(1, batchsize, 1) \\
        返回 padsequence, (L,batchsize,1)
        '''
        outputs, hn = self.gru(x, h0)
        outputs_tensor, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs_tensor
    
    def predict(self, road_ids, start_speed, rawroadfeat:RoadFeatures, training = True):
        if training:
            ret = self.f(road_ids, start_speed, rawroadfeat)
        else:
            with torch.no_grad():
                ret = self.f(road_ids, start_speed, rawroadfeat)
        return ret        
        
    def f(self, road_ids, start_speed, rawroadfeat:RoadFeatures):
        feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
        feat_sequence = feat_sequence.to(self.device)
        start_speed = torch.FloatTensor(start_speed).to(self.device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
        start_speed = start_speed.unsqueeze(0).unsqueeze(2)
        t = self.forward(feat_sequence, start_speed)
        t = torch.sum(t, dim=0).squeeze()   # 变成(batchsize,)
        return t
    

class GRUBoosting:
    '''
    一个非自动化的简单adaboosting，每次训练会训练多一层（创建多一层）简单GRU.
    两个文件夹，一个是GRUonly_res下，有最开始的GRUmodel的参数；一个是GRUBoosting文件夹，下面是若干个simple gru的参数，以及最终预测时候用的 alpha值.
    '''
    def __init__(self, base_grus_dir:str, is_training:bool,
                    input_dim:int, hidden_dim:int, dropout=0.0, device=torch.device('cuda'), training_set_len:int = None, fundGRU:GRUmodel=None) -> None:
        self.alpha_path = os.path.join(base_grus_dir, 'alphas.alpha')   # np.ndarray, 一维.
        try:
            with open(self.alpha_path, "rb") as f:
                self.alphas = pickle.load(f)
        except:
            self.alphas = None
        self.base_dir = base_grus_dir
        base_dir_content = os.listdir(self.base_dir)
        pt_filename = []
        for f in base_dir_content:
            if f.endswith(".pt"):
                pt_filename.append(f)
        pt_filename.sort()
        self.layers = len(pt_filename)

        self.device = device
        self._training_set_len = training_set_len
        self.is_training = is_training
        if is_training:
            self.training_layer = self.layers
            if self.training_layer == 0:
                # 第一层，那么训练样本的分布就是均匀的. 新生成一个.
                self.train_distribution = np.ones((training_set_len), dtype=np.float32) / training_set_len
                with open(os.path.join(self.base_dir, "layer{}.weight".format(self.training_layer)), "wb") as f:
                    pickle.dump(self.train_distribution, f)
                self.train_distribution = torch.from_numpy(self.train_distribution).to(device) * training_set_len
            else:
                # 不是第一层，加载之前计算好的这一层应有的分布.
                with open(os.path.join(self.base_dir, "layer{}.weight".format(self.training_layer)), 'rb') as f:
                    self.train_distribution = pickle.load(f)
                self.train_distribution = torch.from_numpy(self.train_distribution).to(device) * training_set_len
            self.gru2train = SimpleGRU(input_dim, device=device, dropout=dropout).to(device)
            print("GRU boosting initialized. Train the SimpleGRU of layer {}".format(self.training_layer))
        
        else:
            self.alphas = torch.from_numpy(self.alphas).to(device)
            self.alphas /= torch.sum(self.alphas)
            if fundGRU is None:
                raise Exception("Must provide a fundamental GRU model (an instance of GRUmodel classes) when predicting!")
            self.fundGRU = fundGRU
            
            def load_base_gru(param:str, device, input_dim, hidden_size):
                gru = SimpleGRU(input_dim, device).to(device).eval()
                gru.load_state_dict(torch.load(param))
                return gru
            
            self.baseGRUlayers = [load_base_gru(os.path.join(self.base_dir, p), device, input_dim, hidden_dim) for p in pt_filename]
            print("GRU boosting with {} layers initialized in prediction mode.".format(len(self.baseGRUlayers)))

        
    def getDataDistribution(self, indices):
        return self.train_distribution[indices]
    
    def getTrainingParams(self):
        return self.gru2train.parameters()
    
    def getStateDict(self):
        return self.gru2train.state_dict()
    
    def predict(self, road_ids, start_end_points, start_speed, rawroadfeat:RoadFeatures):
        if self.is_training:
            return self.gru2train.predict(road_ids, start_speed, rawroadfeat, True)
        
        else:
            feat_sequence, sequence_lengths = prepare_sequential_packed_features(rawroadfeat, road_ids)
            feat_sequence = feat_sequence.to(self.device)
            start_speed = torch.FloatTensor(start_speed).to(self.device) * 1000 / 60   # 一维. (batchsize,), 换算成m/min
            if self.fundGRU.learn_h0:
                fund_h0 = start_speed.unsqueeze(0).unsqueeze(2)
            else:
                fund_h0 = get_positional_encoding(start_speed, self.fundGRU.hidden_size)   # (batchsize, hiddensize)
                fund_h0[:,0] = start_speed
                fund_h0.unsqueeze_(0)
            mean_speed_per_road, _ = self.fundGRU.forward(feat_sequence, fund_h0)  # mean_speed: (L,T,1), final_speed: (T,1)
            road_lengths_sequence = prepare_road_lengths(rawroadfeat, road_ids, start_end_points[0], start_end_points[1]).to(self.fundGRU.device)  # (L,T,1)
            times = torch.sum(road_lengths_sequence / (mean_speed_per_road + 1e-6), dim=0).squeeze()    # 这样得到的应该是 (T,)

            restimes = [torch.sum(layer.forward(feat_sequence, start_speed.unsqueeze(0).unsqueeze(2)), dim=0).squeeze().unsqueeze(0) for layer in self.baseGRUlayers]
            restimes = torch.cat(restimes, dim=0).to(self.fundGRU.device) * self.alphas.unsqueeze(1)   # 这应该是[n_layers, batchsize]
            return torch.clamp(times + torch.sum(restimes, dim=0), 2.0)    # (T,)
        
    def stop_training(self, label_restimes = None, predicted_restimes = None):
        '''
        label_restimes & predicted_restimes: (dataset_length, )
        停止训练的时候，自动保存参数，并且计算下一次的权重. (如果是在training状态下), 并计算这一层的alpha.
        '''
        if self.is_training:
            self.save_parameter()

            # 计算新的权重.
            delta = label_restimes - predicted_restimes
            real_distribution = self.train_distribution / self._training_set_len
            Et = torch.max(delta)
            eti = delta**2 / Et**2
            sigma_t = torch.sum(real_distribution * eti)
            alpha_t = (sigma_t / (1-sigma_t)).detach().cpu()
            new_distribution = real_distribution * alpha_t ** (1-eti)
            new_distribution /= torch.sum(new_distribution)
            # 将上面的东西保存，也就是alpha和new_distribution.
            if self.training_layer == 0:
                new_alpha = np.array([float(alpha_t)], dtype=np.float32)
            else:
                new_alpha = np.concatenate((self.alphas, np.array([float(alpha_t)], dtype=np.float32)), axis=0)
            with open(self.alpha_path, "wb") as f:
                pickle.dump(new_alpha, f)
            with open(os.path.join(self.base_dir, "layer{}.weight".format(self.training_layer+1)), "wb") as f:
                pickle.dump(new_distribution.detach().cpu().numpy(), f)

    def save_parameter(self):
        if self.is_training:
            torch.save(self.getStateDict(), os.path.join(self.base_dir, "param{}.pt".format(self.training_layer)))



class SimpleGRUBagging(nn.Module):
    def __init__(self, input_dim, device, n_bagging, dropout=0.0, is_training = False, params_path:str = None) -> None:
        super().__init__()
        self.GRUs = [SimpleGRU(input_dim, device, dropout).to(device) for _ in range(n_bagging)]  # 这里就已经有初始化了
        self.device = device
        self.nbags = n_bagging
        self.is_training = is_training
        if not is_training:
            pt_filename = []
            for f in os.listdir(params_path):
                if f.endswith('.pt'):
                    pt_filename.append(f)
            assert len(pt_filename) == n_bagging
            for i in range(n_bagging):
                self.GRUs[i].load_state_dict(torch.load(os.path.join(params_path, pt_filename[i])))
                self.GRUs[i].eval()

    def predict(self, road_ids, start_speed, rawroadfeat:RoadFeatures, id:int = None):
        if self.is_training:
            return self.GRUs[id].predict(road_ids, start_speed, rawroadfeat, True)
        else:
            t = [gru.predict(road_ids, start_speed, rawroadfeat, False).unsqueeze(0) for gru in self.GRUs]
            t = torch.concat(t, 0)
            return torch.sum(t, dim=0) / self.nbags