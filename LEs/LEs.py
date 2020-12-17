import sys
sys.path.append('..')
import torch
import numpy as np

from sklearn.metrics import pairwise_kernels

from utils.datasets import load
from utils.metrics import *


class LELP:

    def __init__(self, alpha=0.5, max_iters=10, n_jobs=None):
        self._a = alpha
        self._nj = n_jobs
        self._e = max_iters
        # attributes
        self.distribution_ = None
    
    def fit(self, X, logical_label):
        # prepare data
        A = pairwise_kernels(X, metric='rbf', n_jobs=self._nj)
        A = A * (np.ones_like(A) - np.eye(A.shape[0]))
        A = torch.from_numpy(A).float()
        L = torch.from_numpy(logical_label).float()
        Dv = torch.diag(1/(torch.sqrt(A.sum(dim=1)) + 1e-15))
        A = Dv.mm(A).mm(Dv)
        D = torch.clone(L)
        # learning

        for i in range(self._e):
            print("iter %d..." % i)
            Dpre = torch.clone(D)
            D = self._a * A.mm(D) + (1 - self._a) * L
            if torch.norm(Dpre - D) < 1e-6:
                break
        self.distribution_ = torch.softmax(D, dim=1)

        # tensor to numpy
        self.distribution_ = self.distribution_.numpy()
        return self
    
    def fit_transform(self, X, logical_label):
        self.fit(X, logical_label)
        return self.distribution_
    
    def score(self, distribution, metric='kldivergence'):
        command = "%s(distribution, self.distribution_)" % metric
        res = eval(command)
        return res

class LEML:
    '''
    parameters:
    K: int, the parameter of KNN, default: 1 + # of labels 
    '''
    def __init__(self, K=None, n_jobs=None):
        self._k = K
        self._nj = n_jobs

    def fit(self, X, logical_label):
        from sklearn.neighbors import kneighbors_graph
        self._k = logical_label.shape[1] + 1
        G = kneighbors_graph(X, n_neighbors=self._k, n_jobs=self._nj)
        
        # learning W
        XX = torch.tensor([X[np.where(G[i].toarray()[0] == 1)] for i in range(X.shape[0])])
        X = torch.from_numpy(X).float()
        W = torch.rand((XX.shape[0], 1, 7), requires_grad=True)
        opter = torch.optim.Adam([W], lr=1e-3)
        lossfn = torch.nn.MSELoss()
        while True:
            opter.zero_grad()
            hat = torch.matmul(W, XX)[:, 0, :]
            loss = lossfn(hat, X)
            loss.backward()
            opter.step()
            if loss.item() < 1e-5:
                break
        
        # learning D
        W = W.detach().numpy()[:,0,:]

        return self

    def score(self, distribution):
        pass
    

# X, ll, ld = load("SJAFFE", return_X_y=True)
# score = LELP(n_jobs=3).fit(X, ll).score(ld, "cosine")
# print(score)

import matplotlib.pyplot as plt
import seaborn as sns

_, _, ld = load("Natural_Scene")
# draw_distribution(ld[:, 0])
fig, axs = plt.subplots(3, 2)

for i in range(3):
    for j in range(2):
        t = i * 2 + j
        sns.histplot(ld[:, t], kde=True, ax=axs[i, j],)
        axs[i, j].set_ylabel('')
plt.show()


# sns.displot(ld[:, 0], bins=80, kde=True)
# plt.show()