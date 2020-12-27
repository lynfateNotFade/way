import sys
sys.path.append("..")

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

import numpy as np
from model import VAE_Encoder, VAE_Bernulli_Decoder

from utils.datasets import load
from utils.metrics import *

X, ll, ld = load("SJAFFE")
X = torch.from_numpy(X).float()
ll = torch.from_numpy(ll).float()
feat_dim, label_dim = X.shape[1], ll.shape[1]
train_data = torch.cat([X, ll], dim=1)

inference = VAE_Encoder(feat_dim + label_dim, 128, label_dim)
generator = VAE_Bernulli_Decoder(label_dim, 128, feat_dim + label_dim)
criteria1 = nn.MSELoss()
criteria2 = binary_cross_entropy
optimizer = torch.optim.Adam(list(inference.parameters()) + list(generator.parameters()), lr=1e-4)

for epoch in range(5000):
    inference.train()
    generator.train()
    optimizer.zero_grad()
    mu, sigma = inference(train_data)
    z = torch.softmax(mu + sigma * torch.randn(mu.shape), dim=1)
    re_train_data = generator(z)
    Xhat, logical_hat = re_train_data[:, :feat_dim], re_train_data[:, feat_dim:]
    loss1 = criteria1(Xhat, X)
    loss2 = criteria2(logical_hat, ll)
    loss = loss1 + loss2
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("epoch %d, loss1: %.2f, loss2: %.2f" % (epoch, loss1.item(), loss2.item()))

inference.eval()
mu, sigma = inference(train_data)
dhat = torch.randn(mu.shape) * sigma + mu
dhat = softmax(dhat.detach().numpy())
print(kldivergence(ld, dhat))
print(chebyshev(ld, dhat))
print(cosine(ld, dhat))