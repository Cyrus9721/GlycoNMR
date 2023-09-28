import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl.nn.pytorch import GraphConv
from dgl import AddSelfLoop


class NMR_GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()

        self.front_dense1 = nn.Linear(in_size, hid_size[0])

        # two-layer GCN
        self.layers.append(
            GraphConv(hid_size[0], hid_size[1], activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size[1], hid_size[2]))
        self.dropout = nn.Dropout(0.5)
        self.back_dense1 = nn.Linear(hid_size[2], hid_size[3])
        self.back_dense2 = nn.Linear(hid_size[3], out_size)

    def forward(self, g, features):
        h = features
        h = self.front_dense1(h)
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        h = self.back_dense1(h)
        h = self.back_dense2(h)
        h = h.view(-1)
        return h
