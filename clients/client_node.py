import torch
from torch import nn

from models.modules import ClientEncoder
from utils.samplers import ClassBalancedBatchSampler


class SplitClient:
    def __init__(self, client_dict, cfg, device):
        self.client_id = client_dict["client_id"]
        self.modality_id = client_dict["modality_id"]
        self.gt_cluster = client_dict["gt_cluster"]
        self.x = client_dict["x"].to(device)
        self.y = client_dict["y"].to(device)
        self.device = device

        self.encoder = ClientEncoder(cfg["input_dim"], cfg["encoder_hidden_dim"]).to(device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=cfg["learning_rate"])
        self.sampler = ClassBalancedBatchSampler(self.y.cpu().tolist(), cfg["batch_size"], rng_seed=cfg["seed"])

    def sample_batch(self):
        idx = self.sampler.sample_indices()
        idx_t = torch.tensor(idx, dtype=torch.long, device=self.device)
        x = self.x[idx_t]
        y = self.y[idx_t]
        return x, y, idx_t

    def forward_to_server(self, x):
        z_client = self.encoder(x)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server

    def backward_from_server(self, z_client, grad_from_server):
        self.optimizer.zero_grad()
        z_client.backward(grad_from_server)
        self.optimizer.step()

    def cluster_representation(self):
        parts = []
        for p in self.encoder.parameters():
            parts.append(p.detach().flatten())
        return torch.cat(parts)
