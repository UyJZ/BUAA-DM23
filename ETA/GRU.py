import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import orthogonal_, constant_, kaiming_normal_

class GRUmodel(nn.Module):
    '''
    当前的设想：用一个单层的GRU完成序列中每个路段上的平均速度的预测. \\
    轨迹构成一个序列，每个路段的输入是当前路段的长度连上它的编码，然后初始隐藏状态时其初始速度v0, 并且扩充成和隐藏层相同维度的张量.. \\
    用一个共享的神经网络(暂时用一个单层的神经网络.)将序列中每个路段的输出结果(是一个和隐藏状态同纬度的东西)变换为这个路段上的平均速度.  \\
    最后的隐藏状态也通过这个神经网络，与轨迹的末尾速度作一个正则化..? \\
    暂时不考虑路口的延迟.
    '''
    def __init__(self, input_dim, hidden_dim, dropout=0.0, learnable_init_hidden_code=False) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, dropout = dropout)
        self.linear = nn.Linear(hidden_dim, hidden_dim//2)
        self.linear1= nn.Linear(hidden_dim//2, 1)

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
            self.h0_linear0 = nn.Linear(1, hidden_dim//2)
            self.h0_linear1 = nn.Linear(hidden_dim//2, hidden_dim)
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