import torch
import torch.nn as nn
import torch.nn.functional as F


class model_ft(torch.nn.Module):
    def __init__(self, ft_in, nb_classes, pretrain_model):
        super(model_ft, self).__init__()
        self.pretrain_model = pretrain_model
        self.fc = nn.Linear(ft_in, nb_classes)

    def forward(self, data):
        x = self.pretrain_model(data)
        x = self.fc(x)
        return x

    # Detach the return variables
    def embed(self, data):
        x = self.pretrain_model(data)
        return x.detach()