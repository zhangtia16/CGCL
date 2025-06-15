import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, global_add_pool # noqa
from torch.nn import Linear 


class GCN(torch.nn.Module):
    dropout_ratio = 0.7
    def __init__(self, args):
        super(GCN, self).__init__()
        num_features = args.num_features
        dim = args.nhid
        self.conv1 = GCNConv(num_features, dim)
        self.conv2 = GCNConv(dim, dim)
        self.conv3 = GCNConv(dim, dim)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.fc1 = Linear(dim, dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()

    def embed(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x.detach()