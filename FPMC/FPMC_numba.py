import sys, os, pickle, time
import math, random
import numpy as np
import numba as nb
from numba import jit, njit
from utils import *
from numba import types

import FPMC as FPMC_basic

class FPMC(FPMC_basic.FPMC):
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular, allowed_trans):
        """
        初始化 FPMC 模型对象

        参数：
        - n_user: 用户数量
        - n_item: 项目数量
        - n_factor: 模型中的因子数量
        - learn_rate: 学习率
        - regular: 正则化参数
        - allowed_trans: 允许的转移集合字典
        """
        super(FPMC, self).__init__(n_user, n_item, n_factor, learn_rate, regular, allowed_trans=allowed_trans)
        self.allowed_trans = allowed_trans

    def evaluation(self, data_3_list):
        """
        评估模型的性能

        参数：
        - data_3_list: 包含三个数据列表的数据，分别是用户列表、项目列表和历史项目列表

        返回值：
        - acc: 准确率
        - mrr: 平均倒数排名
        """
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        #allowed_trans_array_type = types.DictType(types.int32, types.int32[:])
        #allowed_trans_array_numba = allowed_trans_array_type(self.allowed_trans)
        #allowed_trans_array = {key: np.array(value) for key, value in self.allowed_trans.items()}
        #acc, mrr = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU, self.VIL_m_VLI, allowed_trans_array=allowed_trans_array)
        #allowed_trans_keys = np.array(list(self.allowed_trans.keys()))
        #allowed_trans_values = np.array([self.allowed_trans[key][0] if key in self.allowed_trans else -1 for key in allowed_trans_keys])

        #acc, mrr = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU, self.VIL_m_VLI, allowed_trans_keys, allowed_trans_values)
        
        # allowed_train_arrray = np.array(list(self.allowed_trans.values()))
        keys_array = np.array(list(self.allowed_trans.keys()), dtype=int)
        values_list = list(self.allowed_trans.values())
        # print(values_list)
        acc, mrr = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU, self.VIL_m_VLI, values_array=values_list)


        return acc, mrr
   
    def learn_epoch(self, data_3_list, neg_batch_size):
        """
        执行模型的一个学习周期，更新模型参数

        参数：
        - data_3_list: 包含三个数据列表的数据，分别是用户列表、项目列表和历史项目列表
        - neg_batch_size: 负采样的批次大小
        """
        allowed_trans_array = np.array(list(self.allowed_trans))
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             allowed_trans_array, self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=False, ret_in_score=False):
        """
        训练 FPMC 模型

        参数：
        - tr_data: 训练数据
        - te_data: 测试数据（可选）
        - n_epoch: 学习周期数
        - neg_batch_size: 负采样的批次大小
        - eval_per_epoch: 是否每个 epoch 结束后进行评估
        - ret_in_score: 是否返回内部评分（可选）

        返回值：
        - 如果 te_data 不为 None，返回评估指标（acc 和 mrr）；否则，返回 None。
        """
        tr_3_list = data_to_3_list(tr_data)
        if te_data != None:
            te_3_list = data_to_3_list(te_data)

        for epoch in range(n_epoch):

            self.learn_epoch(tr_3_list, neg_batch_size)

            if eval_per_epoch == True:
                acc_in, mrr_in = self.evaluation(tr_3_list)
                if te_data != None:
                    acc_out, mrr_out = self.evaluation(te_3_list)
                    print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
                else:
                    print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
            else:
                print ('epoch %d done' % epoch)

        if eval_per_epoch == False:
            acc_in, mrr_in = self.evaluation(tr_3_list)
            if te_data != None:
                acc_out, mrr_out = self.evaluation(te_3_list)
                print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
            else:
                print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))

        if te_data != None:
            if ret_in_score:
                return (acc_in, mrr_in, acc_out, mrr_out)
            else:
                return (acc_out, mrr_out)
        else:
            return None


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    """
    计算 x 值的 JIT 编译函数

    参数：
    - u: 用户索引
    - i: 项目索引
    - b_tm1: 上一个时间步的历史项目列表
    - VUI: 用户到项目的矩阵
    - VIU: 项目到用户的矩阵
    - VLI: 历史项目到项目的矩阵
    - VIL: 项目到历史项目的矩阵

    返回值：
    - 计算得到的 x 值
    """
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return (np.dot(VUI[u], VIU[i]) + (acc_val/len(b_tm1)))



@jit(nopython=True)
def learn_epoch_jit(u_list, i_list, b_tm1_list, neg_batch_size, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    """
    一个 epoch 的学习的 JIT 编译函数

    参数：
    - u_list: 用户列表
    - i_list: 项目列表
    - b_tm1_list: 上一个时间步的历史项目列表
    - neg_batch_size: 负采样的批次大小
    - item_set: 全部项目的集合
    - VUI: 用户到项目的矩阵
    - VIU: 项目到用户的矩阵
    - VLI: 历史项目到项目的矩阵
    - VIL: 项目到历史项目的矩阵
    - learn_rate: 学习率
    - regular: 正则化参数

    返回值：
    - 更新后的用户和项目矩阵
    """
    for iter_idx in range(len(u_list)):
        d_idx = np.random.randint(0, len(u_list))
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx]!=-1]

        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)

        z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learn_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learn_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm1:
                eta += VLI[l]
            eta = eta / len(b_tm1)

            VILi_update = learn_rate * (delta * eta - regular * VIL[i])
            VILj_update = learn_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm1), VLI.shape[1]))
            for idx, l in enumerate(b_tm1):
                VLI_updates[idx] = learn_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm1)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm1):
                VLI[l] += VLI_updates[idx]

    return VUI, VIU, VLI, VIL


@jit(nopython=True)
def sigmoid_jit(x):
    """
    Sigmoid 函数的 JIT 编译函数

    参数：
    - x: 输入值

    返回值：
    - Sigmoid 函数的计算结果
    """
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))

@jit(nopython=True)
def compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI):
    """
    计算一批数据的 x 值的 JIT 编译函数

    参数：
    - u: 用户索引
    - b_tm1: 上一个时间步的历史项目列表
    - VUI_m_VIU: 用户到项目的矩阵乘积
    - VIL_m_VLI: 历史项目到项目的矩阵乘积

    返回值：
    - 计算得到的一批数据的 x 值
    """
    former = VUI_m_VIU[u]
    latter = np.zeros(VIL_m_VLI.shape[0])
    for idx in range(VIL_m_VLI.shape[0]):
        for l in b_tm1:
            latter[idx] += VIL_m_VLI[idx, l]
    latter = latter/len(b_tm1)

    return (former + latter)

@jit(nopython=True)
def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI, values_array):
    """
    评估模型性能的 JIT 编译函数

    参数：
    - u_list: 用户列表
    - i_list: 项目列表
    - b_tm1_list: 上一个时间步的历史项目列表
    - VUI_m_VIU: 用户到项目的矩阵乘积
    - VIL_m_VLI: 历史项目到项目的矩阵乘积
    - values_array: 某种值的数组

    返回值：
    - 准确率（acc）和平均倒数排名（mrr）
    """
    correct_count = 0
    acc_rr = 0
    for d_idx in range(len(u_list)):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx]!=-1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)


        if i == scores.argmax():
            correct_count += 1

        rank = len(np.where(scores > scores[i])[0]) + 1
        rr = 1.0/rank
        acc_rr += rr

    acc = correct_count / len(u_list)
    mrr = acc_rr / len(u_list)
    return (acc, mrr)
