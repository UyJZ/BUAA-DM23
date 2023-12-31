from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.sparse as sparse
from sklearn.preprocessing import StandardScaler

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 选择GPU的索引，可以根据实际情况修改


from utils import load_data, accuracy, load_data_sparse
from models import GAT, SpGAT

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data() if not args.sparse else load_data_sparse()

print(features.shape)

print(labels.shape)

# features = features.half()
# adj = adj.half()
# labels = labels.half()

if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                  nhid=args.hidden, 
                  nclass=int(labels.max()) + 1,
                  dropout=args.dropout, 
                  nheads=args.nb_heads, 
                  alpha=args.alpha)
else:
    model = GAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)

# 将整个模型参数转换为float16
# model = model.half()

# model = nn.DataParallel(model)

optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda() # 将features也转换为float16
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)

def map_road_node_encodings(node_embeddings):
    scaler = StandardScaler()
    standardized_node_encodings = scaler.fit_transform(node_embeddings)
    road_length = node_embeddings[:, 0]
    intersection_count = node_embeddings[:, 1]  
    road_length_weight = 0.7
    intersection_count_weight = 0.3

    # 将特征进行加权求和，作为新的编码
    weighted_sum_encoding = (road_length_weight * road_length +
                             intersection_count_weight * intersection_count)

    # 将新的编码添加到标准化后的编码中
    mapped_node_encodings = np.column_stack((standardized_node_encodings, weighted_sum_encoding))

    # 你可以根据需要添加更多的自定义映射逻辑

    return mapped_node_encodings


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()

    # 打印每个参数的梯度信息
    #for name, param in model.named_parameters():
    #    if param.grad is not None:
    #        print(f'Gradient - {name}: max={param.grad.max()}, min={param.grad.min()}, mean={param.grad.mean()}')

    optimizer.step()

    if not args.fastmode:
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    if loss_val.data.item() < train.best_loss:
        train.best_loss = loss_val.data.item()
        trained_features = output.cpu().detach().numpy()
        np.save('best_trained_features.npy', trained_features)

    return loss_val.data.item()

# Initialize a variable to keep track of the best loss
'''
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    # 取出 features, adj, labels 从 Variable 中
    features_data = features.data
    adj_data = adj.data
    labels_data = labels.data

    # 将 features 和 adj 的维度调整为符合 DataParallel 要求的形状
    features_data = features_data.view(num_gpus, -1, features_data.size(-1))
    adj_data = adj_data.view(num_gpus, -1, adj_data.size(-2), adj_data.size(-1))

    # 将 labels 的维度调整为符合 DataParallel 要求的形状
    labels_data = labels_data.view(num_gpus, -1)

    # 使用 DataParallel 进行前向传播
    output = model(features_data, adj_data)

    # 将输出的维度调整为符合 DataParallel 要求的形状
    output = output.view(-1, output.size(-1))
    labels_data = labels_data.view(-1)

    # 使用 DataParallel 计算损失
    loss_train = F.nll_loss(output[idx_train], labels_data[idx_train])

    # 使用 DataParallel 进行反向传播和优化器更新
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features_data, adj_data)

    # 将输出的维度调整为符合 DataParallel 要求的形状
    output = output.view(-1, output.size(-1))
    labels_data = labels_data.view(-1)

    # 使用 DataParallel 计算验证集上的损失
    loss_val = F.nll_loss(output[idx_val], labels_data[idx_val])

    acc_train = accuracy(output[idx_train], labels_data[idx_train])
    acc_val = accuracy(output[idx_val], labels_data[idx_val])

    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    # Check if the current model is the best
    if loss_val.data.item() < train.best_loss:
        train.best_loss = loss_val.data.item()

        # Save the trained features of the best model
        trained_features = output.cpu().detach().numpy()
        np.save('best_trained_features.npy', trained_features)

    return loss_val.data.item()
'''

# Initialize a variable to keep track of the best loss
train.best_loss = float('inf')


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    
    with open('test_results.txt', 'a') as f:
        f.write("Epoch {}: Loss={:.4f}, Accuracy={:.4f}\n".format(epoch, loss_test.data.item(), acc_test.data.item()))

# Train model
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))

    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
            
    # torch.cuda.empty_cache()

files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Restore best model
print('Loading {}th epoch'.format(best_epoch))
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# Testing
compute_test()
