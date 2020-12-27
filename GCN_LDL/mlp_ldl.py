import sys
sys.path.append("..")
import numpy as np
from scipy.sparse import coo_matrix
from scipy.stats import entropy

# dgl
import dgl
from dgl.nn.pytorch import GraphConv

# torch
import torch
import torch.nn as nn

# mypackages
from utils.datasets import load
from utils.metrics import *
from utils.transforms import *

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_kernels
from sklearn.preprocessing import OneHotEncoder

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, out_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(x, dim=1)

X, _, Y = load('Natural_Scene')

for j in range(1, 10):
    print("---------%d-------" % j)
    
    
    label_embeds = OneHotEncoder().fit_transform(np.arange(Yr.shape[1]).reshape(-1, 1)).toarray()
    label_graph = dgl.from_scipy(construct_ldl_label_graph(Yr))

    # to tensor
    Xr = torch.from_numpy(Xr).float()
    Yr = torch.from_numpy(Yr).float()
    Xs = torch.from_numpy(Xs).float()
    Ys = torch.from_numpy(Ys).float()
    label_embeds = torch.from_numpy(label_embeds).float()

    model = MLGCN(Xr.shape[1], label_embeds.shape[1])
    # model = MLP(Xr.shape[1], Yr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    for epoch in range(3000):
        optimizer.zero_grad()
        Ypred = model(Xr, label_embeds, label_graph)
        loss = torch.sum(torch.pow(Ypred - Yr, 2))  
        # Ypred = model(Xr)
        # loss = torch.sum(torch.pow(Ypred - Yr, 2))
        
        loss.backward()
        optimizer.step()
        if epoch % 1000 == 0:
            print(epoch)
            # Yspred = mlgcn(Xs, label_embeds, label_graph).detach().numpy()
            # Yrpred = Ypred.detach().numpy()
            # Yrreal = Yr.numpy()
            # Ysreal = Ys.numpy()
            # s = "-------iter %4d-------\n" + \
            #     "train/test KL:       %.4f, %.4f\n" + \
            #     "train/test Cheb:     %.4f, %.4f\n" + \
            #     "train/test intersec: %.4f, %.4f\n" + \
            #     "train/test Cosine:   %.4f, %.4f\n"
            # print(s % (epoch, 
            #     kldivergence(Yrreal, Yrpred), kldivergence(Ysreal, Yspred),
            #     chebyshev(Yrreal, Yrpred), chebyshev(Ysreal, Yspred),
            #     intersection(Yrreal, Yrpred), intersection(Ysreal, Yspred),
            #     cosine(Yrreal, Yrpred), cosine(Ysreal, Yspred),
            #     ))

    s = "KL:       %.4f\n" + \
        "Cheb:     %.4f\n" + \
        "intersec: %.4f\n" + \
        "Cosine:   %.4f\n"
    Ysreal = Ys.numpy()
    Yspred = model(Xs, label_embeds, label_graph).detach().numpy()
    # Yspred = model(Xs).detach().numpy()
    print(s % (kldivergence(Ysreal, Yspred), chebyshev(Ysreal, Yspred), intersection(Ysreal, Yspred), cosine(Ysreal, Yspred),))