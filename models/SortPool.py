import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import SAGEConv, global_sort_pool


class SortPool(torch.nn.Module):
    def __init__(self, args):
        super(SortPool, self).__init__()
        num_features = args.num_features
        hidden = args.nhid
        num_layers = 2
        self.dropout_ratio = args.dropout_ratio
        self.k = 30
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.conv1d = Conv1d(hidden, 32, 5)
        self.lin1 = Linear(32 * (self.k - 5 + 1), hidden)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.conv1d.reset_parameters()
        self.lin1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout_ratio, training=self.training)
            x = F.relu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = global_sort_pool(x, batch, self.k)
        x = x.view(len(x), self.k, -1).permute(0, 2, 1)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.conv1d(x))
        x = x.view(len(x), -1)
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin1(x))

        return x
    def __repr__(self):
        return self.__class__.__name__
