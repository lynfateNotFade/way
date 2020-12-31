import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F

def gauss_kl_loss(mu,sigma,eps = 1e-12):
    mu_square = torch.pow(mu,2)
    sigma_square = torch.pow(sigma,2)
    loss = mu_square + sigma_square - torch.log(eps+sigma_square) - 1
    loss = 0.5 * loss.mean(1)
    return loss.mean()

class VariationalAutoEncoder(nn.Module):
    
    def __init__(self, in_dim, hid_dim, drop_prob=0.0,):
        '''
        in_dim: the dimension of data space.
        hid_dim: arbitrary
        forword function returns (mu, sigma), estimated_data
        '''
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



def variational_density_estimate(X):
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X)
    loader = Data.DataLoader(
        dataset=Data.TensorDataset(X),
        batch_size=100,
        shuffle=True
    )
    vae = VariationalAutoEncoder(X.shape[1], 128)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    loss_series = []
    for epoch in range(300):
        loss = 0
        for batch_id, (x,) in enumerate(loader):
            optimizer.zero_grad()
            (mu, sigma), xhat = vae(x)
            gauss_loss = gauss_kl_loss(mu, sigma)
            rec_loss = F.mse_loss(xhat, x)
            loss = 0.5 * gauss_loss + rec_loss
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            loss_series.append(round(loss.item(), 3))

    (mu, sigma), _ = vae(X)
    print(loss_series)
    return mu, sigma