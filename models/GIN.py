# -*- coding: utf-8 -*-


import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool, global_max_pool 

class GIN(torch.nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()

        num_features = args.num_features
        dim = args.nhid
        self.dropout_ratio = args.dropout_ratio

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)

    def forward(self, dataset):
        x = dataset.x
        edge_index = dataset.edge_index
        batch = dataset.batch
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.bn4(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
#        x = F.dropout(x, p=0.5, training=self.training)
#        x = self.fc2(x)
#        return F.log_softmax(x, dim=-1)
        return x

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv4.reset_parameters()
        self.conv5.reset_parameters()
        self.fc1.reset_parameters()

    # Detach the return variables
    def embed(self, dataset):
        x = dataset.x
        edge_index = dataset.edge_index
        batch = dataset.batch
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.bn4(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x.detach()