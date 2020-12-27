import sys
sys.path.append('..')
import torch
import torch.nn as nn
from utils.datasets import load
from utils.metrics import *


X, ll, ld = load("Yeast_alpha")

X = torch.from_numpy(X).float()
ll = torch.from_numpy(ll).float()
ld = torch.from_numpy(ld).float()
C = ll.shape[1]
A = torch.rand((C, C), requires_grad=True)


optimizer = torch.optim.Adam([A], lr=1e-5)
criteria = nn.MSELoss()
for i in range(ll.shape[0]):
    for epoch in range(20000):
        optimizer.zero_grad()
        hat = torch.softmax(ll[[i]].mm(A.mm(A.t())), dim=1)
        loss = criteria(hat, ld[[i]])
        loss.backward()
        optimizer.step()
    print(loss.item())
