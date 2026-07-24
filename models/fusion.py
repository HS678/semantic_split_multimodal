import torch
from torch import nn

from models.modules import ClusterAdapter


class ConcatMLPFusionServer(nn.Module):
    def __init__(self, cluster_ids, feature_dim, num_classes, cfg, cluster_to_slot=None):
        super().__init__()
        self.cluster_ids = sorted(int(cluster_id) for cluster_id in cluster_ids)
        if not self.cluster_ids:
            raise ValueError("ConcatMLPFusionServer requires at least one cluster slot.")
        if cluster_to_slot is None:
            cluster_to_slot = {cluster_id: slot for slot, cluster_id in enumerate(self.cluster_ids)}
        self.cluster_to_slot = {int(cluster_id): int(slot) for cluster_id, slot in cluster_to_slot.items()}
        expected_slots = list(range(len(self.cluster_ids)))
        if sorted(self.cluster_to_slot) != self.cluster_ids or sorted(self.cluster_to_slot.values()) != expected_slots:
            raise ValueError("cluster_to_slot must map every pred_cluster id to a contiguous fusion slot.")
        self.slot_to_cluster = {
            slot: cluster_id
            for cluster_id, slot in self.cluster_to_slot.items()
        }

        fusion_cfg = cfg.get("fusion", {})
        server_cfg = cfg.get("model", {}).get("server", {})
        adapter_dim = int(fusion_cfg.get("adapter_dim", server_cfg.get("adapter_dim", feature_dim)))
        hidden_dim = int(fusion_cfg.get("hidden_dim", server_cfg.get("hidden_dim", adapter_dim)))
        dropout = float(fusion_cfg.get("dropout", server_cfg.get("dropout", 0.0)))
        num_layers = max(1, int(fusion_cfg.get("num_layers", 1)))

        self.adapters = nn.ModuleDict(
            {
                str(cluster_id): ClusterAdapter(
                    feature_dim,
                    adapter_dim,
                    hidden_dim=server_cfg.get("adapter_hidden_dim"),
                    dropout=dropout,
                )
                for cluster_id in self.cluster_ids
            }
        )

        layers = []
        current_dim = adapter_dim * len(self.cluster_ids)
        for _ in range(num_layers):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, int(num_classes)))
        self.classifier = nn.Sequential(*layers)

    def forward(self, slot_activations: dict[int, torch.Tensor]):
        missing = [cluster_id for cluster_id in self.cluster_ids if cluster_id not in slot_activations]
        if missing:
            raise ValueError(f"Missing fusion slot activations for pred_cluster ids: {missing}")

        adapted = []
        batch_size = None
        for slot in range(len(self.cluster_ids)):
            cluster_id = self.slot_to_cluster[slot]
            activation = slot_activations[cluster_id]
            if batch_size is None:
                batch_size = int(activation.shape[0])
            elif int(activation.shape[0]) != batch_size:
                raise ValueError("All fusion slot activations must share the same batch size.")
            adapted.append(self.adapters[str(cluster_id)](activation))

        fused = torch.cat(adapted, dim=1)
        return self.classifier(fused), fused
