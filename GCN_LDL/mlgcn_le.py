import sys
sys.path.append("..")
import numpy as np

# dgl
import dgl

# torch
import torch
import torch.nn as nn
# mypackages
from utils.datasets import load
from utils.metrics import *

from mlgcn import MLGCN

# sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


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




X, Y, YD = load("Yeast_alpha")
Xr, Xs, Yr, Ys, YDr, YDs = train_test_split(X, Y, YD, random_state=0)
label_embeds = PCA(n_components=8).fit_transform(Yr.T)
# label_embeds = Yr.copy().T
label_graph = dgl.from_scipy(construct_label_graph(Yr, is_reweight=False))

# to tensor
Xr = torch.from_numpy(Xr).float()
Yr = torch.from_numpy(Yr).long()
label_embeds = torch.from_numpy(label_embeds).float()

mlgcn = MLGCN(Xr.shape[1], label_embeds.shape[1])
lossfns = [nn.CrossEntropyLoss() for _ in range(Yr.shape[1])]
optimizer = torch.optim.Adam(mlgcn.parameters(), lr=5e-3)
uniform = torch.ones_like(Yr).float() / Yr.shape[1]

for epoch in range(5000):
    optimizer.zero_grad()
    Ypred = mlgcn(Xr, label_embeds, label_graph)
    loss_sum = 0
    for i in range(Yr.shape[1]):
        P = Ypred[:, i].reshape(-1, 1)
        Yp = torch.cat([1-P, P], dim=1)
        loss_sum += lossfns[i](Yp, Yr[:, i])
    # Ypred_smx = torch.softmax(Ypred, dim=1)
    # loss_sum += 1e-2 * torch.norm(uniform - Ypred_smx)    
    loss_sum.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(epoch, loss_sum.item(), "kl: ", 
            kldivergence(YDr, torch.softmax(Ypred.detach(), dim=1).numpy())
        )

Ypred = torch.softmax(mlgcn(Xr, label_embeds, label_graph), dim=1)

