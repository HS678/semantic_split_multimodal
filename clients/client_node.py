import torch
import torch.nn.functional as F

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
        self.cfg = cfg
        self.input_dim = int(client_dict.get("input_dim", cfg.get("input_dim", self.x.shape[1])))

        self.encoder = ClientEncoder(self.input_dim, cfg["encoder_hidden_dim"]).to(device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=cfg["learning_rate"])
        self.sampler = ClassBalancedBatchSampler(self.y.cpu().tolist(), cfg["batch_size"], rng_seed=cfg["seed"])

        self._stage1_pretrained = False

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

    # alias for protocol tests / readability
    def backward_update(self, z_client, grad_from_server):
        self.backward_from_server(z_client, grad_from_server)

    def pretrain_autoencoder(
        self,
        enabled=True,
        epochs=5,
        batch_size=64,
        lr=0.001,
        weight_decay=0.0,
        max_samples=None,
    ):
        if not enabled:
            return {"enabled": False, "epochs": 0, "avg_recon_loss": None}
        if self._stage1_pretrained:
            return {"enabled": True, "epochs": 0, "avg_recon_loss": None}

        prev_mode = self.encoder.training
        self.encoder.train()

        x_all = self.x
        if max_samples is not None:
            n = min(int(max_samples), int(x_all.shape[0]))
            perm = torch.randperm(int(x_all.shape[0]), device=self.device)[:n]
            x_all = x_all[perm]

        decoder = torch.nn.Linear(int(self.cfg["encoder_hidden_dim"]), self.input_dim).to(self.device)
        ae_opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(decoder.parameters()),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )
        mse = torch.nn.MSELoss()

        bs = max(1, int(batch_size))
        total_loss = 0.0
        total_batches = 0

        for _ in range(max(0, int(epochs))):
            perm = torch.randperm(int(x_all.shape[0]), device=self.device)
            for start in range(0, int(x_all.shape[0]), bs):
                idx = perm[start : start + bs]
                xb = x_all[idx]

                z = self.encoder(xb)
                x_hat = decoder(z)
                loss = mse(x_hat, xb)

                ae_opt.zero_grad()
                loss.backward()
                ae_opt.step()

                total_loss += float(loss.item())
                total_batches += 1

        if prev_mode:
            self.encoder.train()
        else:
            self.encoder.eval()

        self._stage1_pretrained = True
        avg_loss = (total_loss / total_batches) if total_batches > 0 else None
        return {"enabled": True, "epochs": int(epochs), "avg_recon_loss": avg_loss}

    def cluster_representation(self, max_samples=None, normalize=False):
        prev_mode = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            x_all = self.x
            if max_samples is not None:
                n = min(int(max_samples), int(x_all.shape[0]))
                x_all = x_all[:n]
            z = self.encoder(x_all)
            mean_z = z.mean(dim=0)
            std_z = z.std(dim=0, unbiased=False)
            fingerprint = torch.cat([mean_z, std_z], dim=0)
            if normalize:
                fingerprint = F.normalize(fingerprint.unsqueeze(0), p=2, dim=1).squeeze(0)
        if prev_mode:
            self.encoder.train()
        return fingerprint.detach()
