import torch
import torch.nn as nn
import torch.nn.functional as F


class GIN_ft(torch.nn.Module):
    def __init__(self, ft_in, nb_classes, GIN):
        super(GIN_ft, self).__init__()
        self.GIN = GIN
        self.fc = nn.Linear(ft_in, nb_classes)

    def forward(self, data):
        x = self.GIN(data)
        x = self.fc(x)
        return x

    # Detach the return variables
    def embed(self, data):
        x = self.GIN(data)
        return x.detach()