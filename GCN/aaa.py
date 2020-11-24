import sys
sys.path.append('..')
from utils.datasets import load
from utils.metrics import *

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_kernels

from skorch import NeuralNetRegressor


# class Net(nn.Module):
#     def __init__(self, in_dim, hid_dim, out_dim, L):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hid_dim)
#         self.fc2 = nn.Linear(hid_dim, out_dim)
#         self.fc3 = nn.Linear(out_dim, out_dim)
#         self.act1 = nn.LeakyReLU()
#         self.sfx = nn.Softmax(dim=1)
#         self.L = L

#     def forward(self, x):
#         x = self.act1(self.fc1(x))
#         x = self.act1(self.fc2(x))
#         x = x.mm(self.L)
#         x = self.sfx(self.fc3(x))
#         return x

# class Net(nn.Module):
#     def __init__(self, in_dim, out_dim, L):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, out_dim)
#         self.act1 = nn.LeakyReLU()
#         self.fc2 = nn.Linear(out_dim, out_dim)
#         self.sfx = nn.Softmax(dim=1)
#         self.L = L

#     def forward(self, x):
#         x = self.act1(self.fc1(x))
#         # x = x.mm(self.L)
#         x = self.sfx(self.fc2(x))
#         return x

class Net(nn.Module):
    def __init__(self, in_dim, out_dim, L):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(out_dim, out_dim)
        self.sfx = nn.Softmax(dim=1)
        self.L = L

    def forward(self, x):
        # x = self.fc1(x)
        # x = x.mm(self.L)
        x = self.sfx(self.fc1(x))
        return x

torch.manual_seed(0)

X, y, _ = load("medical", return_X_y=True, problem='mll')
# print(y.shape)
for line in y:
    print(np.where(line == 1))

# s = 0
# for i in range(5):
#     Xr, Xs, yr, ys = train_test_split(X, y, random_state=i)
#     L = pairwise_kernels(yr.T, metric='linear')
#     L = L / L.sum(axis=1).reshape(-1, 1)

#     # print(L)
#     reg = NeuralNetRegressor(
#         module=Net,
#         optimizer=torch.optim.Adam,
#         # optimizer__weight_decay=5e-8,
#         criterion=nn.MSELoss,
#         module__in_dim=X.shape[1],
#         # module__hid_dim=128,
#         module__out_dim=y.shape[1],
#         module__L = torch.from_numpy(L).float(),
#         max_epochs=100,
#         lr=1e-2,
#         train_split=None,
#     )
#     reg.fit(Xr, yr)
#     yhat = reg.predict(Xs)
#     s += intersection(ys, yhat)
# print("inte: ", s/5)
