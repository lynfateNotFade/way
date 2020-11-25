import sys
sys.path.append("..")
import numpy as np
# dgl
import dgl
from dgl.nn.pytorch import GraphConv
# torch
import torch
import torch.nn as nn
# mypackages
from utils.datasets import load
# sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hid_dim)
        self.gc2 = GraphConv(hid_dim, out_dim)
    
    def forward(self, G, x):
        x = self.gc1(G, x)
        x = torch.relu(x)
        x = self.gc2(G, x)
        return x

class MLGCN(nn.Module):
    def __init__(self, ins_feat_dim, label_embeds_dim):
        super().__init__()
        in_dim, out_dim = label_embeds_dim, ins_feat_dim
        hid_dim = int(in_dim / 2 + out_dim / 2)
        self.gc1 = GraphConv(in_dim, hid_dim)
        self.gc2 = GraphConv(hid_dim, out_dim)
    
    def forward(self, X, label_embeds, label_graph):
        h = self.gc1(label_graph, label_embeds)
        h = torch.relu(h)
        h = self.gc2(label_graph, h).t()
        return torch.sigmoid(X.mm(h))

def get_label_cooccur(label_matrix:np.ndarray, n_jobs=None):
    '''
    Return a matrix M whose element M_{ij} denotes 
    the concurring times of label i and label j.
    '''
    from sklearn.metrics import pairwise_kernels
    counts_i_j = lambda yi, yj: yi[np.logical_and(yi == 1, yj == 1)].size
    return pairwise_kernels(label_matrix.T, metric=counts_i_j, n_jobs=n_jobs)

def construct_label_graph(label_matrix:np.ndarray, 
                        cooccur_threshold=0.4, 
                        is_reweight=True,
                        self_label_weight=0.2,
                        n_jobs=None):
    '''
    self_label_weight is required if is_reweight is True.
    '''
    from scipy.sparse import coo_matrix
    M = get_label_cooccur(label_matrix, n_jobs=n_jobs)
    N = np.array([col[col > 0].size for col in label_matrix.T]).reshape(-1, 1)
    P = M / N
    P[P >= cooccur_threshold] = 1.
    P[P < cooccur_threshold] = 0
    
    if not is_reweight:
        return coo_matrix(P)
    A_zero_diag = P * (np.ones_like(P) - np.eye(P.shape[0]))


X, _, Y = load("SJAFFE", return_X_y=True)
Xr, Xs, Yr, Ys = train_test_split(X, Y, random_state=0)
label_embeds = PCA(n_components=32).fit_transform(Yr.T)
label_graph = dgl.from_scipy(construct_label_graph(Yr, is_reweight=False))

# to tensor
Xr = torch.from_numpy(Xr).float()
Yr = torch.from_numpy(Yr).long()
label_embeds = torch.from_numpy(label_embeds).float()

mlgcn = MLGCN(Xr.shape[1], label_embeds.shape[1])
lossfns = [nn.CrossEntropyLoss() for _ in range(Yr.shape[1])]
optimizer = torch.optim.Adam(mlgcn.parameters(), lr=1e-4)

for epoch in range(1000):
    optimizer.zero_grad()
    Ypred = mlgcn(Xr, label_embeds, label_graph)
    loss_sum = 0
    for i in range(Yr.shape[1]):
        P = Ypred[:, i].reshape(-1, 1)
        Yp = torch.cat([1-P, P], dim=1)
        loss_sum += lossfns[i](Yp, Yr[:, i])
    loss_sum.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(loss_sum.item())