import sys
sys.path.append("..")

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
from utils.datasets import load
from utils.estimators import NeuralLDRegressor
from utils.metrics import *

from sklearn.cluster import *
from sklearn.model_selection import train_test_split
'''
We treat logical labels as the samples of a latent population-level label distribution.
'''

def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(sigma,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()

class VariationalAutoEncoder(nn.Module):
    
    def __init__(self, feat_dim, label_dim, hid_dim, drop_prob=0.0,):
        super().__init__()
        in_dim = feat_dim + label_dim
        self.inference_ = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.ELU(), nn.Dropout(drop_prob),
            nn.Linear(hid_dim, hid_dim), nn.Tanh(), nn.Dropout(drop_prob),
            nn.Linear(hid_dim, in_dim * 2)
        )
        self.generator_ = nn.Sequential(
            nn.Linear(in_dim, hid_dim), nn.Tanh(), nn.Dropout(drop_prob),
            nn.Linear(hid_dim, hid_dim), nn.ELU(), nn.Dropout(drop_prob),
            nn.Linear(hid_dim, in_dim * 2)
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.01)
    
    def forward(self, data, feat_dim, label_dim):
        in_dim = feat_dim + label_dim
        latent = self.inference_(data)
        feat, label = latent[:, :feat_dim * 2], latent[:, feat_dim * 2:]
        feat_mu, feat_sigma = feat[:, :feat_dim], feat[:, feat_dim:]
        label_mu, label_sigma = label[:, :label_dim], label[:, label_dim:]
        label_mu = torch.softmax(label_mu, dim=1)
        feat = feat_mu + torch.randn(feat_mu.shape) * feat_sigma
        label = label_mu + torch.randn(label_mu.shape) * label_sigma
        
        rec = self.generator_(torch.cat([feat, label], dim=1))
        feat, label = rec[:, :feat_dim * 2], rec[:, feat_dim * 2:]
        _feat_mu, _feat_sigma = feat[:, :feat_dim], feat[:, feat_dim:]
        _label_mu, _label_sigma = label[:, :label_dim], label[:, label_dim:]
        _label_mu = torch.softmax(_label_mu, dim=1)
        feat = _feat_mu + torch.randn(_feat_mu.shape) * _feat_sigma
        label = _label_mu + torch.randn(_label_mu.shape) * _label_sigma
        return (feat_mu, feat_sigma), (label_mu, label_sigma), torch.cat([feat, label], dim=1)


def variational_density_estimate(X, feat_dim, label_dim):
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    loader = Data.DataLoader(
        dataset=Data.TensorDataset(X),
        batch_size=100,
        shuffle=True
    )
    vae = VariationalAutoEncoder(feat_dim, label_dim, 128)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(900):
        loss = 0
        for batch_id, (x,) in enumerate(loader):
            optimizer.zero_grad()
            (feat_mu, feat_sigma), (label_mu, label_sigma), xhat = vae(x, feat_dim, label_dim)
            gauss_loss1 = gauss_kl_loss(feat_mu, feat_sigma)
            gauss_loss2 = gauss_kl_loss(label_mu, label_sigma)
            gauss_loss = gauss_loss1 + gauss_loss2
            rec_loss = F.mse_loss(xhat, x)
            loss = 0.5 * gauss_loss + rec_loss
            loss.backward()
            optimizer.step()
    
    grad_sum = 0
    for param in list(vae.parameters()):
        if len(param) != 2:
            continue
        W, b = list(layer.parameters())
        grad_sum += (W.grad.sum() + b.grad.sum()) 

    (feat_mu, feat_sigma), (label_mu, label_sigma), _ = vae(X, feat_dim, label_dim)
    print("[VariationalAutoEncoder is trained] Gradient sum: %.5f" % grad_sum)
    return (feat_mu.detach(), feat_sigma.detach()), (label_mu.detach(), label_sigma.detach())

X, ll, ld = load("SBU_3DFE")
X, Xs, ld, ls = train_test_split(X, ld, test_size=0.9)
print("training set contains %d instances.\ntest set contains %d instances" % (X.shape[0], Xs.shape[0]))

est = NeuralLDRegressor(epoches=2000).fit(X, ld)
print(cosine(ls, est.predict(Xs)))
print(kldivergence(ls, est.predict(Xs)))

divider = KMeans(n_clusters=ld.shape[1] * 3, random_state=0).fit(X)
cluster_ids = divider.labels_
Xadd, Yadd = [], []

for clu_id in np.unique(cluster_ids):
    data = np.concatenate([
        ld[cluster_ids == clu_id],
        X[cluster_ids == clu_id]
    ], axis=1)
    (feat_mu, feat_sigma), (label_mu, label_sigma) = variational_density_estimate(data, X.shape[1], ld.shape[1])
    for _ in range(5):
        Xadd.append(feat_mu + feat_sigma * torch.randn(feat_mu.shape))
        Yadd.append(label_mu + label_sigma * torch.randn(label_mu.shape))
Xadd = torch.cat(Xadd, dim=0).numpy()
Yadd = torch.cat(Yadd, dim=0).numpy()
X = np.concatenate([Xadd, X], axis=0)
ld = np.concatenate([Yadd, ld], axis=0)
print(X.shape)
print(ld.shape)
est = NeuralLDRegressor(epoches=2000).fit(X, ld)
print(cosine(ls, est.predict(Xs)))
print(kldivergence(ls, est.predict(Xs)))