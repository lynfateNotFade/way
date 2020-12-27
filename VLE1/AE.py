import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.datasets import load
from utils.metrics import *

class AutoEncoder(nn.Module):
    
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, latent_dim), nn.Softmax(dim=1)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, in_dim // 2), nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim)
        )
    def forward(self, x):
        latent = self.enc(x)
        return latent, self.dec(latent)

# create data
X, ll, ld = load("SJAFFE")
X = torch.from_numpy(X).float()
ll = torch.from_numpy(ll).float()
n_feat, n_label = X.shape[1], ll.shape[1]

# create model
xae = AutoEncoder(n_feat, n_label)
yae = AutoEncoder(n_label, n_label)
params = list(xae.parameters()) + list(yae.parameters())
optimizer = torch.optim.SGD(params, lr=1e-4)

# train model
xae.train()
yae.train()
for epoch in range(10000):
    optimizer.zero_grad()
    # forward
    dist_xae, dec_feat = xae(X)
    dist_yae, dec_label = yae(ll)

    # loss
    xae_loss = F.mse_loss(dec_feat, X)
    yae_loss = F.mse_loss(dec_label, ll)
    diff_loss = torch.norm(dist_xae - dist_yae)
    dist_loss = F.kl_div(dist_yae, ll)
    loss = xae_loss + yae_loss + diff_loss + dist_loss
    # backward
    loss.backward()
    optimizer.step()
    # print
    if epoch % 100 == 0:
        print(cosine(ld, dist_yae.detach().numpy()))
