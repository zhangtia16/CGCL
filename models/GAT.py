import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, global_add_pool, global_max_pool 
from torch.nn import Linear 

class GAT(torch.nn.Module):
    dropout_ratio = 0.6
    def __init__(self, args):
        super(GAT, self).__init__()
        num_features = args.num_features
        dim = args.nhid
        heads = 2
        self.conv1 = GATConv(num_features, dim, heads=heads, dropout=self.dropout_ratio)
        self.conv2 = GATConv(dim*heads, dim, heads=heads, dropout=self.dropout_ratio)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv3 = GATConv(dim*heads, dim, heads=1, dropout=self.dropout_ratio)
        self.fc1 = Linear(dim, dim)

    def forward(self,data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        batch = data.batch
        x = F.dropout(data.x, p=self.dropout_ratio, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.elu(self.conv2(x, data.edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.conv3(x, data.edge_index)
        x = F.dropout(x, p =self.dropout_ratio, training=self.training)
        
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        return x



