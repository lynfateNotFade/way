# dgl
from dgl.nn.pytorch import GraphConv
# torch
import torch
import torch.nn as nn

class MLGCN(nn.Module):
    def __init__(self, ins_feat_dim, label_embeds_dim):
        super().__init__()
        in_dim, out_dim = label_embeds_dim, ins_feat_dim
        self.gc1 = GraphConv(in_dim, 32)
        self.gc2 = GraphConv(32, 16)
        self.gc3 = GraphConv(16, out_dim)
        # self.fc = nn.Linear()
    
    def forward(self, X, label_embeds, label_graph):
        h = torch.relu(self.gc1(label_graph, label_embeds))
        h = torch.relu(self.gc2(label_graph, h))
        h = torch.relu(self.gc3(label_graph, h)).t()
        return torch.sigmoid(X.mm(h))