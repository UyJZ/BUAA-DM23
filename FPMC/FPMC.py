import sys, os, pickle, time
import math, random
import numpy as np
from utils import *
from tqdm import tqdm
from scipy.special import expit

class FPMC():
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular, allowed_trans):
        """
        初始化 FPMC 模型的参数和属性。

        参数:
        - n_user: 用户数目
        - n_item: 项目数目
        - n_factor: 模型的潜在因子数量
        - learn_rate: 学习率
        - regular: 正则化项系数
        - allowed_trans: 允许的转移状态字典
        """
        self.user_set = set()
        self.item_set = set()

        self.n_user = n_user
        self.n_item = n_item

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular
        self.allowed_trans = allowed_trans

    @staticmethod
    def dump(fpmcObj, fname):
        """
        将 FPMC 模型保存到文件。

        参数:
        - fpmcObj: FPMC 模型对象
        - fname: 文件名
        """
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        """
        从文件中加载 FPMC 模型。

        参数:
        - fname: 文件名

        返回:
        - FPMC 模型对象
        """
        return pickle.load(open(fname, 'rb'))

    def init_model(self, std=0.01):
        """
        初始化模型参数。

        参数:
        - std: 随机初始化时的标准差
        """
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def compute_x(self, u, i, b_tm1):
        """
        计算单个数据点的 x 值。

        参数:
        - u: 用户索引
        - i: 项目索引
        - b_tm1: 之前的项目状态列表

        返回:
        - 计算得到的 x 值
        """
        #acc_val = 0.0
        #for l in b_tm1:
        #    acc_val += np.dot(self.VIL[i], self.VLI[l])
        #return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val/len(b_tm1)))
        acc_val = 0.0
        for l in b_tm1:
            if l in self.allowed_trans[i]:
                acc_val += np.dot(self.VIL[i], self.VLI[l])
        return (np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1)))
        #user_embedding = self.VUI[u]
        #current_item_embedding = self.VIU[i]
        #allowed_b_tm1 = [l for l in b_tm1 if l in self.allowed_trans[i]]
        # 计算当前项目到下一个项目的转移嵌入向量（假设转移嵌入矩阵为 VIL）
        #transition_embedding = np.mean([self.VIL[l] for l in allowed_b_tm1], axis=0)
        # 计算用户对当前项目到下一个项目的兴趣分数
        #score = np.dot(user_embedding, current_item_embedding) + np.dot(current_item_embedding, transition_embedding)
        #return score

    def compute_x_batch(self, u, b_tm1):
        """
        计算一批数据点的 x 值。

        参数:
        - u: 用户索引
        - b_tm1: 之前的项目状态列表

        返回:
        - 计算得到的 x 值
        """
        #former = self.VUI_m_VIU[u]
        
        ## 获取当前状态允许的状态转移列表
        #current_state = b_tm1[-1]
        #allowed_transitions = self.allowed_trans[current_state]
        #print('current state : ', current_state, ' allowed_transitions : ', allowed_transitions)
        
        ## 仅考虑允许的转移状态
        #allowed_b_tm1 = [l for l in b_tm1 if l in allowed_transitions]
        
        ## 计算后半部分
        #latter = np.mean(self.VIL_m_VLI[:, allowed_b_tm1], axis=1).T
        #return (former + latter)
        #former = self.VUI_m_VIU[u]
        #possible_transitions = [self.allowed_trans[i] for i in b_tm1]
        #possible_transitions_flat = [item for sublist in possible_transitions for item in sublist]
        #possible_transitions_set = set(possible_transitions_flat)
        #possible_transitions_set.discard(u)
        #latter = np.mean(self.VIL_m_VLI[:, list(possible_transitions_set)], axis=1).T
        #return (former + latter)
        former = self.VUI_m_VIU[u]
        try:
            current_state = b_tm1[-1]
            allowed_transitions = self.allowed_trans[current_state]
            allowed_b_tm1 = [l for l in b_tm1 if l in allowed_transitions]
        except:
            allowed_b_tm1 = list(range(len(self.allowed_trans)))
        
        latter = np.mean(self.VIL_m_VLI[:, allowed_b_tm1], axis=1).T
        # latter[~np.isin(range(len(self.VIL_m_VLI)), allowed_b_tm1)] = -1

        return (former + latter)


    '''
    def evaluation(self, data_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        correct_count = 0
        rr_sum = 0.0
        for (u, i, b_tm1) in data_list:
            #print(b_tm1)
            scores = self.compute_x_batch(u, b_tm1)

            i_score = scores[i]
            ranks = np.argsort(scores)[::-1]
            rank = np.where(ranks == i)[0][0] + 1
            rr = 1.0 / rank
            rr_sum += rr

            if i_score == max(scores):
                correct_count += 1

        try:
            acc = correct_count / len(data_list)
            mrr = rr_sum / len(data_list)
            return (acc, mrr)
        except ZeroDivisionError:
            return (0.0, 0.0)
    '''
    '''
    def evaluation(self, data_list):
        """
        评估模型在给定数据集上的性能。

        参数:
        - data_list: 数据集，包含用户、项目、之前的项目状态的三元组列表

        返回:
        - 准确率和平均倒数排名元组 (acc, mrr)
        """
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        correct_count = 0
        rr_sum = 0.0

        for (u, i, b_tm1) in tqdm(data_list, desc="Evaluating", unit="tuple", leave=False):
            scores = self.compute_x_batch(u, b_tm1)

            i_score = scores[i]
            ranks = np.argsort(scores)[::-1]
            rank = np.where(ranks == i)[0][0] + 1
            rr = 1.0 / rank
            rr_sum += rr

            if i_score == max(scores):
                correct_count += 1

        try:
            acc = correct_count / len(data_list)
            mrr = rr_sum / len(data_list)
            return (acc, mrr)
        except ZeroDivisionError:
            return (0.0, 0.0)
    '''
    def evaluation(self, data_list):
        """
        评估模型在给定数据集上的性能。

        参数:
        - data_list: 数据集，包含用户、项目、之前的项目状态的三元组列表

        返回:
        - 准确率和平均倒数排名元组 (acc, mrr)
        """
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        correct_count = 0
        rr_sum = 0.0

        return_result = []

        for (u, i, b_tm1, traj_id) in tqdm(data_list, desc="Evaluating", unit="tuple", leave=False):
            scores = self.compute_x_batch(u, b_tm1)
            if i == 6777:
                print(b_tm1)
            # ranks = np.argsort(scores)[::-1]
            rank = len(np.where(scores > scores[i])[0]) + 1
            # print(len(scores))
            rr = 1.0 / rank
            rr_sum += rr
            s = self.allowed_trans[b_tm1[-1]]
            separator = ';'
            str2 = separator.join(str(num) for num in s)

            str1 = "entity_id: " + str(u) + " traj_id: " + str(traj_id) + " expected: " + str(i) + " predict: " + str(s[scores[s].argmax()]) + ' current status: ' + str(b_tm1[-1]) + ' allowed_status: ' + str2 + '\n'
            return_result.append(str1)
            
            
            if i == s[scores[s].argmax()]:
                correct_count += 1

        try:
            acc = correct_count / len(data_list)
            mrr = rr_sum / len(data_list)
            return (acc, mrr,return_result)
        except ZeroDivisionError:
            return (0.0, 0.0,return_result)


    def learn_epoch(self, tr_data, neg_batch_size):
        """
        进行一轮训练。

        参数:
        - tr_data: 训练数据集，包含用户、项目、之前的项目状态的三元组列表
        - neg_batch_size: 负样本批次大小
        """
        for iter_idx in range(len(tr_data)):
            (u, i, b_tm1, traj_id) = random.choice(tr_data)
            #print('u : ', u)
            #print('allowed_exclu_set :' ,allowed_exclu_set)
            #print('neg_batch_size :', neg_batch_size)        
            #if len(self.allowed_trans[u]) >= neg_batch_size:
            #    j_list = random.sample(self.allowed_trans[u], neg_batch_size)
            #else:
                # 处理 neg_batch_size 大于可用集合的情况
            #    j_list = random.sample(self.allowed_trans[u], len(self.allowed_trans[u]))
            # 从最新状态的可达状态中选择负样本
            try:
                latest_state = b_tm1[-1]
                s = self.allowed_trans[latest_state]
            except:
                s = list(range(len(self.allowed_trans)))
            if len(s) >= neg_batch_size:
                j_list = random.sample(s, neg_batch_size)
            else:
                j_list = random.sample(s, len(s))
            if i in j_list:
                j_list.remove(i)
            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                self.VUI[u] += VUI_update
                self.VIU[i] += VIUi_update
                self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * ((delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update
                self.VLI[b_tm1] += VLI_update
                
    '''
    def learn_epoch(self, tr_data, neg_batch_size):
        """
        进行一轮训练。

        参数:
        - tr_data: 训练数据集，包含用户、项目、之前的项目状态的三元组列表
        - neg_batch_size: 负样本批次大小
        """
        # 将用户的可用转移状态数量存储为numpy数组，以便后续使用
        allowed_trans_counts = np.array([len(self.allowed_trans[i]) for i in range(self.n_user)])

        # 使用tqdm包装tr_data，创建一个进度条
        for iter_idx in tqdm(range(len(tr_data)), desc="Training Epoch", unit="iteration"):
            (u, i, b_tm1) = random.choice(tr_data)

            # 从最新状态的可达状态中选择负样本
            latest_state = i
            neg_indices = np.random.choice(self.allowed_trans[latest_state], size=neg_batch_size, replace=True)

            z1 = self.compute_x(u, i, b_tm1)

            # 计算正样本对应的分数
            i_score = self.compute_x(u, i, b_tm1)

            # 计算负样本对应的分数
            j_scores = self.compute_x_batch(u, b_tm1)

            # 计算差异
            deltas = 1 - expit(i_score - j_scores)

            # 计算梯度更新
            VUI_update = self.learn_rate * (deltas[:, np.newaxis] * (self.VIU[i] - self.VIU[neg_indices]) - self.regular * self.VUI[u])
            VIUi_update = self.learn_rate * (deltas[:, np.newaxis] * self.VUI[u] - self.regular * self.VIU[i])
            VIUj_update = self.learn_rate * (-deltas[:, np.newaxis] * self.VUI[u] - self.regular * self.VIU[neg_indices])

            # 更新用户和项目嵌入
            self.VUI[u] += np.sum(VUI_update, axis=0)
            self.VIU[i] += VIUi_update.sum(axis=0)
            self.VIU[neg_indices] += VIUj_update.sum(axis=0)

            # 更新用户的平均嵌入
            eta = np.mean(self.VLI[b_tm1], axis=0)
            VILi_update = self.learn_rate * (deltas[:, np.newaxis] * eta - self.regular * self.VIL[i])
            VILj_update = self.learn_rate * (-deltas[:, np.newaxis] * eta - self.regular * self.VIL[neg_indices])
            VLI_update = self.learn_rate * (
                (deltas[:, np.newaxis] * (self.VIL[i] - self.VIL[neg_indices]) / len(b_tm1)) - self.regular * self.VLI[b_tm1]
            )

            # 更新项目嵌入和用户的平均嵌入
            self.VIL[i] += VILi_update.sum(axis=0)
            self.VIL[neg_indices] += VILj_update.sum(axis=0)
            self.VLI[b_tm1] += VLI_update.sum(axis=0)
    '''
    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=False):
        """
        训练 FPMC 模型并可选地在每个轮次进行评估。

        参数:
        - tr_data: 训练数据集，包含用户、项目、之前的项目状态的三元组列表
        - te_data: 测试数据集，可选
        - n_epoch: 训练轮次
        - neg_batch_size: 负样本批次大小
        - eval_per_epoch: 是否在每个轮次结束后进行评估，可选
        """
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)

            if eval_per_epoch == True:
                acc_in, mrr_in, _ = self.evaluation(tr_data)
                if te_data is not None:
                    acc_out, mrr_out, _ = self.evaluation(te_data)
                    print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
                else:
                    print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))
            else:
                print ('epoch %d done' % epoch)

        if eval_per_epoch == False:
            print('evaluating~~~~')
            acc_in, mrr_in, return_result = self.evaluation(tr_data)
            with open("tr.txt",'w') as f :
                for str1 in return_result:
                    f.write(str1)
                f.close()
            if te_data is not None:
                acc_out, mrr_out, return_result = self.evaluation(te_data)
                with open("te.txt",'w') as f :
                    for str1 in return_result:
                        f.write(str1)
                    f.close()
                print ('In sample:%.4f\t%.4f \t Out sample:%.4f\t%.4f' % (acc_in, mrr_in, acc_out, mrr_out))
            else:
                print ('In sample:%.4f\t%.4f' % (acc_in, mrr_in))

        if te_data is not None:
            return (acc_out, mrr_out)
        else:
            return None
