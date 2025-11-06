# coding=utf-8
import numpy as np
from sklearn.metrics import pairwise_distances
import scipy.sparse as sparse

from hyperg.hyperg import HyperG
from hyperg.utils import print_log
import torch
from scipy.sparse import csr_matrix




def Entropy(input_,class_num):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log2(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)

    c = torch.log2(torch.tensor(class_num))

    w1 = entropy / c

    w = torch.exp(w1)

    return w


def gen_knn_hg(X, hoop_bank,score_bank,n_neighbors,class_num, is_prob=True, with_feature=False):
     
    #:param X: numpy array, shape = (n_samples, n_features)
    #:param n_neighbors: int,
    #:param is_prob: bool, optional(default=True)
    #:param with_feature: bool, optional(default=False)
    #:return: instance of HyperG
    X_cpu = X.cpu()
    X = X_cpu.numpy()
    assert isinstance(X, (np.ndarray, list))
    assert n_neighbors > 0

    X = np.array(X)
    n_nodes = X.shape[0]
    n_edges = n_nodes

    m_dist = pairwise_distances(X,metric='cosine')

    # top n_neighbors+1
    m_neighbors = np.argpartition(m_dist, kth=n_neighbors+1, axis=1)
    m_neighbors_val = np.take_along_axis(m_dist, m_neighbors, axis=1)

    m_neighbors = m_neighbors[:, :n_neighbors+1]
    m_neighbors_val = m_neighbors_val[:, :n_neighbors+1]
    
    
    # check
    for i in range(n_nodes):
        if not np.any(m_neighbors[i, :] == i):
            m_neighbors[i, -1] = i
            m_neighbors_val[i, -1] = 0.
    
          
    m_neighbors1 = m_neighbors[:, :-1]
    
    neighbor_scores = score_bank[m_neighbors1]
    
    
    epsilon = 1e-10
    neighbor_scores = torch.clamp(neighbor_scores, epsilon, 1.0 - epsilon)

    average_scores = torch.mean(neighbor_scores, dim=1)
    hoops = Entropy(average_scores,class_num)
    hoop_bank = hoops.detach().clone()
    hoop_bank_cpu = hoop_bank.cpu()
    hoop_bank = hoop_bank_cpu.numpy()
    diag_hoop = np.diag(hoop_bank)   

    node_idx = m_neighbors.reshape(-1)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)

    if not is_prob:
        values = np.ones(node_idx.shape[0])
    else:
        avg_dist = np.mean(m_dist)
        m_neighbors_val = m_neighbors_val.reshape(-1)
        values = np.exp(-np.power(m_neighbors_val, 2.) / np.power(avg_dist, 2.))

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    w = np.ones(n_edges)
    H = H + csr_matrix(diag_hoop)
    if with_feature:
        return HyperG(H, w=w, X=X)

    return HyperG(H, w=w)
    
