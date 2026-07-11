import torch
from torch import nn
from models.encoders import MLPEncoder


ClientEncoder = MLPEncoder


class SemanticProjector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        return self.net(x)


class ConcatFusion(nn.Module):
    def forward(self, projected_list):
        return torch.cat(projected_list, dim=1)


class AttentionFusion(nn.Module):
    def forward(self, projected_list):
        raise NotImplementedError("AttentionFusion is a v1 stub.")


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
