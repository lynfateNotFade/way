import torch
import torch.nn as nn

class NeuralLDRegressor:
    
    def __init__(self, net=None, criteria=None, optimizer=None, epoches=300, lr=1e-3):
        self._n = net
        self._c = criteria
        self._o = optimizer
        self._e = epoches
        self._l = lr

        self.grad_sum_ = 0

    def fit(self, X, Y):
        X = X if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
        Y = Y if isinstance(Y, torch.Tensor) else torch.from_numpy(Y).float()
        if self._n is None:
            self._n = nn.Sequential(
                nn.Linear(X.shape[1], X.shape[1] * 2),
                nn.ReLU(),
                nn.Linear(X.shape[1] * 2, Y.shape[1]),
                nn.Softmax(dim=1),
            )
        criteria = nn.MSELoss() if self._c is None else self._c
        optimizer = torch.optim.Adam(self._n.parameters(), lr=self._l) if self._o is None else self._o
        
        for epoch in range(self._e):
            optimizer.zero_grad()
            Yhat = self._n(X)
            loss = criteria(Yhat, Y)
            loss.backward()
            optimizer.step()
        
        grad_sum = 0
        for param in list(self._n.parameters()):
            if len(param) != 2:
                continue
            W, b = list(param)
            grad_sum += (W.grad.sum() + b.grad.sum()) 
        self.grad_sum_ = grad_sum

        print("[NeuralLDRegressor is trained] Gradient sum: %.5f" % grad_sum)
        return self
    
    def predict(self, X):
        X = X if isinstance(X, torch.Tensor) else torch.from_numpy(X).float()
        return self._n(X).detach().numpy()
