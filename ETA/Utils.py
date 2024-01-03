import numpy as np
import torch
from Dataset import RoadFeatures, TrajDatasetNoGraph
#from GRU import GRUmodel, GRUBoosting

def prepare_sequential_packed_features(road_feat: RoadFeatures, road_ids: list[np.ndarray]):
    '''
    road_ids是列表，里面每一项都是一个轨迹所经过的路段编号，这个编号是一维的np.ndarray, (dtype=int)  \\
    with_road_length: 如果是，就在最前面加上轨迹的长度..
    返回： \\
        PackedSequence: (L, T, n_features) L是最长的序列，T是batch_size, n_features是特征的维度. \\
        sequence_length: (T,) batch里每个序列的长度，是一个list.
    '''
    sequence_length = list(map(lambda x:len(x), road_ids))
    L = max(sequence_length)
    N = len(road_ids)
    padded_sequence = np.zeros((L,N, road_feat.n_features), dtype=np.float32)
    for i in range(N):
        l = sequence_length[i]
        ids = road_ids[i]
        feats = road_feat.getFeatures(ids)
        # if with_road_length:
        #     road_lengths = road_feat.getRoadLength(ids)  # 一维数组.
        #     padded_sequence[:l, i, :] = np.concatenate((road_lengths[:,np.newaxis], feats), axis=1)
        # else:
        padded_sequence[:l, i, :] = feats
    #print(padded_sequence[0,0,:])
    packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(torch.from_numpy(padded_sequence), sequence_length, enforce_sorted=False)
    return packed_sequence, sequence_length


def prepare_road_lengths(road_feat: RoadFeatures, road_ids: list[np.ndarray], start_matched_point:np.ndarray, final_matched_point:np.ndarray):
    '''
    返回每个路段的长度，注意，第一条路和最后一条路会只算到匹配的起始点或终点位置.  \\
    start_matched_point和final_matched_point都应该是np一维数组(长度为2).
    返回： \\
        sequence_length: (L, T, 1), L是sequence(最长的)的长度，T是batchsize，1是路段长度. (tensor)
    '''
    # 所用的路网纬度大约40度，x轴(经度)度数差对应的距离大概 111000*cos40 约85031m，y轴度数差(维度)对应距离大概111000m.
    # x_unit = 85031   # 85031m/°
    # y_unit = 111000
    sequence_length = list(map(lambda x:len(x), road_ids))
    L = max(sequence_length)
    N = len(road_ids)
    lengths = np.zeros((L,N,1), dtype=np.float32)
    for i in range(N):
        ids = road_ids[i]
        l = len(ids)
        if start_matched_point is not None and final_matched_point is not None:
            if l==1:
                # 起点终点在同一条路上.. 暂时直接用起点终点直线距离吧..虽然路段中有折线所以这样不对，但因为路段都比较短所以还好..?
                dist = computeApproxDistFromCoord(start_matched_point, final_matched_point)  # 单位m.
                lengths[0,i,0] = dist
            else:
                this_sequence_lengths = road_feat.getRoadLength(ids)  # 1维
                end_of_first_road = road_feat.getRoadTarget(ids[0])  # 1维
                start_of_last_road= road_feat.getRoadOrigin(ids[-1]) # 1维
                this_sequence_lengths[0] = computeApproxDistFromCoord(start_matched_point, end_of_first_road)
                this_sequence_lengths[-1]= computeApproxDistFromCoord(final_matched_point, start_of_last_road)
                lengths[:l,i,0] = this_sequence_lengths
    return torch.from_numpy(lengths)


__unit = np.array([85301, 111000])
def computeApproxDistFromCoord(point1:np.ndarray, point2:np.ndarray):
    return np.sqrt(np.sum(((point1-point2)*__unit)**2))



def get_positional_encoding(x, hidden_size, exponential:int = None):
    '''
    sin, cos轮流上，按照transformer中的公式，PE(x, 2i)=sin(x/(10000^(2i/d_model))), PE(x, 2i+1)=cos(x/(10000^(2i/d_model))) \\
    输入： x, 一维tensor(长度为batch_size). hidden_size: int; \\
    输出： (batchsize, hiddensize);
    '''
    ret = torch.ones((x.shape[0], hidden_size)).to(x.device)  # (batch_size, hidden_size)
    if exponential is None:
        i = torch.arange(hidden_size).to(x.device)
        div = (10000 ** (i/hidden_size)).to(x.device)  # (hidden_size)

        _x = x.unsqueeze(1)   # 变成 (batch_size, 1)
        inner = _x / div      # 变成 (batch_size, hidden_size)，能在长度为1的维度上广播.

        ret[:,0::2] = torch.sin(inner[:,0::2])
        ret[:,1::2] = torch.cos(inner[:,0::2])
    else:
        i = torch.arange(hidden_size // 2).to(x.device)
        inner = (exponential ** i) * (x.unsqueeze(1))
        ret[:,0::2] = torch.sin(inner)
        ret[:,1::2] = torch.cos(inner)
    return ret


def prepare_input(hour, holiday, start_speed, n_hidden_hour, n_hidden_holiday, n_hidden_speeds):
    h00 = get_positional_encoding(hour, n_hidden_hour, exponential=2)
    h01 = torch.ones((holiday.shape[0], n_hidden_holiday)).to(holiday.device) * holiday.unsqueeze(1)
    h02 = get_positional_encoding(start_speed, n_hidden_speeds)   # (batchsize, hiddensize//3)
    h02[:,0] = start_speed
    return torch.cat([h00, h01, h02], dim=1).unsqueeze(0)   # (1, batchsize, hiddensize)


def predict_all_trajs(trajSet: TrajDatasetNoGraph, model, rawroadfeat:RoadFeatures, label:torch.Tensor = None):
    '''
    返回(len(trajSet), 2)，[:,0]是label, [:,1]是预测值. 如果没有提供label, 就用trajSet中的 duration, 即轨迹的真实时间
    '''
    l = 0
    b = 8
    with torch.no_grad():
        ret = torch.zeros((len(trajSet), 2), dtype=torch.float32).to(model.device)
        for batch in trajSet.iter_traj_by_order(b):
            if trajSet.hour_holiday:
                traj_info, road_ids, duration, start_speed, final_speed, hour, holiday = batch
            else:
                traj_info, road_ids, duration, start_speed, final_speed = batch
            if hour is not None and holiday is not None:
                try:
                    t = model.predict(road_ids, traj_info, start_speed, rawroadfeat, hour, holiday)
                except:
                    # 不是基于GRUmodel的
                    t = model.predict(road_ids, start_speed, rawroadfeat)
            else:
                try:
                    t = model.predict(road_ids, traj_info, start_speed, rawroadfeat)
                except:
                    t = model.predict(road_ids, start_speed, rawroadfeat)
            if label is not None:
                ret[l:l+b,0] = label[l:l+b]
            else:
                ret[l:l+b,0] = torch.tensor(duration).to(model.device)
            ret[l:l+b,1] = t
            l += start_speed.shape[0]    
    return ret