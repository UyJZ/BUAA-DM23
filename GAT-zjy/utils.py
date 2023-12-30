import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import os

cwd = os.path.dirname(os.path.realpath(__file__))
data = os.path.join(cwd, '..', 'database', 'data')
eta_path = os.path.join(cwd, '..', 'database', 'ETA')
rel_path = os.path.join(data, 'rel.csv')

proportion_train = 0.7
proportion_val = 0.2
proportion_test = 0.1


def encode_onehot_sparse(labels):
    classes = sorted(list(set(labels)))
    classes_dict = {c: i for i, c in enumerate(classes)}
    labels_indices = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    num_classes = len(classes)
    labels_onehot = sp.coo_matrix((np.ones_like(labels_indices), (np.arange(len(labels_indices)), labels_indices)), shape=(len(labels), num_classes), dtype=np.int32)
    return labels_onehot

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize_adj(mx):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize feature matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def load_data_sparse():
    path = os.path.join(eta_path, 'road_features.csv')
    print('Loading {} dataset...', path)

    # Load features and labels
    df = pd.read_csv(path)
    # Extract header
    header = df.columns
    # Extract coordinates
    coordinates = df['coordinates'].apply(eval).values
    # Extract features excluding ID and coordinates
    features = sp.csr_matrix(df.iloc[:, 2:].values.astype(np.float32))
    labels = encode_onehot_sparse(df['id'].values.astype(np.int32))
    edges_df = pd.read_csv(rel_path)
    edges = edges_df[['origin_id', 'destination_id']].values

    # Build graph adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize features and adjacency matrix
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    trainCount = int(labels.shape[0] * proportion_train)
    valCount = int(labels.shape[0] * proportion_val)
    testCount = int(labels.shape[0] * proportion_test)
    idx_train = range(trainCount)
    idx_val = range(trainCount, trainCount + valCount)
    idx_test = range(trainCount + valCount, trainCount + valCount + testCount)

    adj_coo = torch.sparse_coo_tensor(torch.LongTensor(np.vstack(adj.nonzero())),
                                    torch.FloatTensor(adj.data),
                                    torch.Size(adj.shape))

    # 创建稀疏特征矩阵
    features_coo = torch.sparse_coo_tensor(torch.LongTensor(np.vstack(features.nonzero())),
                                        torch.FloatTensor(features.data),
                                        torch.Size(features.shape))

    # 转换标签为 LongTensor
    labels = torch.LongTensor(labels.nonzero()[1])


    # 转换索引为 LongTensor
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return adj_coo, features_coo, labels, idx_train, idx_val, idx_test

def load_data():
    path = os.path.join(eta_path, 'road_features.csv')
    print('Loading {} dataset...'.format(path))

    # Load features and labels
    df = pd.read_csv(path)
    df = df.sample(frac=0.02, random_state=42)
    # Extract features excluding ID and coordinates
    features = sp.csr_matrix(df.iloc[:, 2:].values.astype(np.float32))
    labels = encode_onehot(df['id'].values.astype(np.int32))
    print('Success to build')

    edges_df = pd.read_csv(rel_path)
    edges = edges_df[['origin_id', 'destination_id']].values

    # Filter edges based on existing nodes in labels
    existing_nodes = set(labels.nonzero()[1])
    filter_condition = np.logical_and(np.isin(edges[:, 0], list(existing_nodes)),
                                       np.isin(edges[:, 1], list(existing_nodes)))
    edges = edges[filter_condition]

    # Build graph adjacency matrix
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize features and adjacency matrix
    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    trainCount = int(labels.shape[0] * proportion_train)
    valCount = int(labels.shape[0] * proportion_val)
    testCount = int(labels.shape[0] * proportion_test)
    idx_train = range(trainCount)
    idx_val = range(trainCount, trainCount + valCount)
    idx_test = range(trainCount + valCount, trainCount + valCount + testCount)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test