import sys
sys.path.append("..")

import torch.nn as nn
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_kernels

from utils.metrics import *
from utils.datasets import load
from utils.estimators import NeuralLDRegressor

# '''
# transform the label distribution to relative importance.
# [(y1, y1), (y1, y2), (y1, y3), (y1, y4)]
# '''

X, ll, ld = load("SJAFFE")
R = np.array([pairwise_kernels(r.reshape(-1, 1), metric=lambda a,b:b/a) for r in ld])
Y = R.reshape(-1, ld.shape[1] * ld.shape[1])

als, bls = [], []
for _ in range(50):
    print(_)
    Xr, Xs, Yr, Ys, Lr, Ls = train_test_split(X, ld, Y)
    model = NeuralLDRegressor(net=nn.Sequential(nn.Linear(X.shape[1], ld.shape[1]),), epoches=300)
    model.fit(Xr, Yr)
    print(model.grad_sum_)
    Yhat = model.predict(Xs)
    me = cosine(Ys, softmax(Yhat))
    als.append(me)

    model = NeuralLDRegressor(net=nn.Sequential(nn.Linear(X.shape[1], Y.shape[1])), epoches=3000)
    model.fit(Xr, Lr)
    print(model.grad_sum_)
    Yhat = model.predict(Xs)
    R = Yhat.reshape(-1, ld.shape[1], ld.shape[1])
    Yhat = np.array([softmax(np.real(np.linalg.eigh(r)[0])) for r in R])
    me = cosine(Ys, Yhat)
    bls.append(me)
    break
als, bls = np.array(als), np.array(bls)
print(np.mean(als))
print(np.mean(bls))