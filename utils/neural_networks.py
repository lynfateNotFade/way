import torch
import torch.nn as nn

def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(sigma,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()

class VariationalAutoEncoder(nn.Module):
    
    def __init__(self, in_dim, hid_dim, drop_prob=0.0,):
        super().__init__()
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
    
    def forward(self, X):
        in_dim = X.shape[1]
        mean_var = self.inference_(X)
        mu, sigma = mean_var[:, :in_dim], mean_var[:, in_dim:]
        X = mu + torch.randn(mu.shape) * sigma
        mean_var = self.generator_(X)
        mean, var = mean_var[:, :in_dim], mean_var[:, in_dim:]
        X = mean + torch.randn(mean.shape) * var
        return (mu, sigma), X


from datasets import load
import torch.utils.data as Data
import torch.nn.functional as F

X, ll, ld = load("SJAFFE")
X = torch.from_numpy(X).float()
loader = Data.DataLoader(
    dataset=Data.TensorDataset(X), 
    batch_size=100,
    shuffle=True
    )

vae = VariationalAutoEncoder(X.shape[1], 128)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
for epoch in range(100):
    for batch_id, (x,) in enumerate(loader):
        optimizer.zero_grad()
        (mu, sigma), xhat = vae(x)
        gauss_loss = gauss_kl_loss(mu, sigma)
        rec_loss = F.mse_loss(xhat, x)
        loss = 0.5 * gauss_loss + rec_loss
        loss.backward()
        optimizer.step()

(mu, sigma), _ = vae(X)

sample = torch.randn(mu.shape) * sigma + mu
print(sample.shape)