import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.datasets import load
from utils.metrics import *

class Infer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.map_for_mean = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.Softmax(dim=1)
        )
        self.map_for_var = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    
    def forward(self,inputs):
        mean = self.map_for_mean(inputs)
        std = self.map_for_var(inputs)
        return mean, std

class Bernulli_Generate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.Sigmoid()
        )
    def forward(self, latent_data):
        return self.mlp(latent_data)

class Gauss_Generate(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.map_for_mean = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.map_for_var = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, latent_data):
        mean = self.map_for_mean(latent_data)
        std = self.map_for_var(latent_data)
        return mean, std

def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(mu,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()

# create data
X, ll, ld = load("Yeast_alpha")
X = torch.from_numpy(X).float()
ll = torch.from_numpy(ll).float()
data = torch.cat((X, ll), 1)
n_feat, n_label = X.shape[1], ll.shape[1]
# create model
enc = Infer(in_dim=n_feat+n_label, out_dim=n_label)
x_dec = Gauss_Generate(in_dim=n_label, out_dim=n_feat)
y_dec = Bernulli_Generate(in_dim=n_label, out_dim=n_label)
param_list = list(enc.parameters())+list(x_dec.parameters())+list(y_dec.parameters())
optimizer = torch.optim.Adam(param_list, lr=1e-3)

# training
enc.train()
x_dec.train()
y_dec.train()

latent_true_diff = []
label_true_diff = []
latent_shape_loss = []
kls = []

# ----------------------------training----------------------------------
for epoch in range(4000):
    optimizer.zero_grad()
    # forward
    (mu, sigma) = enc(data)
    z = mu + sigma * torch.randn(mu.size())
    mean, var = x_dec(z)
    xhat = mean + var * (torch.randn(mean.size()))
    yhat = y_dec(z)
    d = torch.softmax(z, dim=1)
    # loss
    rec_loss_x = F.mse_loss(xhat, X)
    kl_loss = gauss_kl_loss(mu, sigma)
    rec_loss_y1 = F.binary_cross_entropy(mu, ll)
    rec_loss_y2 = F.binary_cross_entropy(yhat, ll)
    loss = 1 * kl_loss + 1 * rec_loss_x + 1 * rec_loss_y1 + 7 * rec_loss_y2
    # backward
    loss.backward()
    optimizer.step()

    pred = kldivergence(ld, mu.detach().numpy())
    latent_true_diff.append(rec_loss_y1)
    latent_shape_loss.append(kl_loss)
    label_true_diff.append(rec_loss_y2)
    kls.append(pred)
    # print
    if epoch % 50 == 0:
        print("epoch %d, loss: %.3f, label_dec_loss: %.3f, latent_loss: %.3f, kl: %.3f"
         % (epoch, loss.item(), rec_loss_y1 + rec_loss_y2, kl_loss, pred))

import matplotlib.pyplot as plt
plt.plot(range(len(kls)), kls, 'r')
plt.plot(range(len(kls)), latent_shape_loss, 'b')
plt.plot(range(len(kls)), latent_true_diff, 'g')
plt.plot(range(len(kls)), label_true_diff, 'pink')
plt.show()

# evaluation
enc.eval()
mu, sigma = enc(data)
ldhat = mu + sigma * torch.randn(mu.shape)
ldhat = torch.softmax(ldhat, dim=1)
print(cosine(ld, ldhat.detach().numpy()))