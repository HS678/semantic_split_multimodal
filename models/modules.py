from torch import nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(int(in_dim), int(num_classes))

    def forward(self, x):
        return self.fc(x)


class ClusterAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim or out_dim)
        layers = [nn.Linear(int(in_dim), hidden_dim), nn.ReLU()]
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(hidden_dim, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedSemanticBackbone(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, num_layers=1, dropout=0.0):
        super().__init__()
        out_dim = int(out_dim or in_dim)
        hidden_dim = int(hidden_dim or out_dim)
        layers = []
        current_dim = int(in_dim)
        for _ in range(max(1, int(num_layers))):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, x):
        return self.net(x)
