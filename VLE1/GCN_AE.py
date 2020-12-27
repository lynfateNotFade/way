import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import numpy as np

from sklearn.neighbors import radius_neighbors_graph

from utils.datasets import load
from utils.metrics import *

class GraphAutoEncoder(nn.Module):
    
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, in_dim // 2)
        self.conv2 = GCNConv(in_dim // 2, in_dim // 4)
        self.enc = nn.Sequential(
            nn.Linear(in_dim // 4, latent_dim), nn.Softmax(dim=1)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim)
        )
    def forward(self, x, edge_index):
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        latent = self.enc(x)
        return latent, self.dec(latent)

# create data
X, ll, ld = load("SJAFFE")
S = radius_neighbors_graph(ll, radius=0.9, include_self=True)
# print(S[S == 1].flatten().size / 212 / 212)
edgelist = []
for i in range(S.shape[0]):
    a = S[i].indices.reshape(1, -1)
    a = np.concatenate([a, np.ones_like(a) * i], axis=0)
    edgelist.append(a)
edgelist = torch.from_numpy(np.concatenate(edgelist, axis=1)).long()
X = torch.from_numpy(X).float()
ll = torch.from_numpy(ll).float()
data = torch.cat([X, ll], 1)
n_feat, n_label = X.shape[1], ll.shape[1]

# create model
gae = GraphAutoEncoder(n_feat + n_label, n_label)
optimizer = torch.optim.Adam(gae.parameters(), lr=1e-4)

# train model
gae.train()
for epoch in range(500):
    optimizer.zero_grad()
    # forward
    dist, dec_val = gae(data, edgelist)
    # loss
    dec_loss = F.mse_loss(dec_val, data)
    dist_loss = F.mse_loss(dist, ll)
    loss = dec_loss + dist_loss
    # backward
    loss.backward()
    optimizer.step()
    # print
    if epoch % 10 == 0:
        print(cosine(ld, dist.detach().numpy()))
