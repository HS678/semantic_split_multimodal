import math

import torch
from torch import nn


class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, conv_channels=(64, 128), kernel_sizes=(5, 3), dropout=0.0):
        super().__init__()
        if input_shape is None:
            raise ValueError("TimeSeriesEncoder requires input_shape.")
        if len(input_shape) == 1:
            input_shape = [1, int(input_shape[0])]
        if len(input_shape) != 2:
            raise ValueError(f"TimeSeriesEncoder expects [channels, length], got {input_shape}")
        self.input_shape = [int(input_shape[0]), int(input_shape[1])]
        c1, c2 = [int(v) for v in conv_channels]
        k1, k2 = [int(v) for v in kernel_sizes]
        self.net = nn.Sequential(
            nn.Conv1d(self.input_shape[0], c1, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c2, int(hidden_dim)),
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            channels, length = self.input_shape
            x = x.reshape(x.shape[0], channels, length)
        return self.net(x)


class ImageEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, pretrained=False):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception as exc:
            raise ImportError("ImageEncoder requires torchvision for the ResNet18 interface.") from exc
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.model = resnet18(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, int(hidden_dim))

    def forward(self, x, lengths=None):
        return self.model(x)


class VideoEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super().__init__()
        in_channels = int(input_shape[0]) if input_shape else 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, int(hidden_dim)),
        )

    def forward(self, x, lengths=None):
        return self.net(x)


class AudioEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super().__init__()
        if input_shape is None:
            input_shape = [1, 128]
        if len(input_shape) == 1:
            input_shape = [1, int(input_shape[0])]
        self.input_shape = [int(input_shape[0]), int(input_shape[1])]
        self.net = nn.Sequential(
            nn.Conv1d(self.input_shape[0], 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, int(hidden_dim)),
        )

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.reshape(x.shape[0], self.input_shape[0], self.input_shape[1])
        return self.net(x)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

    def forward(self, x, lengths=None):
        return self.net(x.reshape(x.shape[0], -1))


def _masked_sequence_mean(x, lengths):
    if lengths is None:
        return x.mean(dim=1)
    lengths = lengths.to(device=x.device, dtype=torch.long).clamp(min=1, max=x.shape[1])
    mask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    weighted = x * mask.unsqueeze(-1).to(dtype=x.dtype)
    return weighted.sum(dim=1) / lengths.unsqueeze(1).to(dtype=x.dtype)


class GRUSequenceEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, num_layers=1, dropout=0.1, bidirectional=False):
        super().__init__()
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"GRUSequenceEncoder expects [time, feature_dim], got {input_shape}")
        input_dim = int(input_shape[1])
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=bool(bidirectional),
        )
        gru_output_dim = self.hidden_dim * (2 if bool(bidirectional) else 1)
        self.output_projection = (
            nn.Identity() if gru_output_dim == self.hidden_dim else nn.Linear(gru_output_dim, self.hidden_dim)
        )

    def forward(self, x, lengths=None):
        x = x.float()
        if x.ndim != 3:
            raise ValueError(f"GRUSequenceEncoder expects [batch, time, feature], got {tuple(x.shape)}")
        output, _ = self.gru(x)
        return self.output_projection(_masked_sequence_mean(output, lengths))


class ConvGRUSequenceEncoder(nn.Module):
    def __init__(self, input_shape, hidden_dim, conv_channels=(32, 64, 128), kernel_size=5, dropout=0.1):
        super().__init__()
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"ConvGRUSequenceEncoder expects [time, feature_dim], got {input_shape}")
        feature_dim = int(input_shape[1])
        channels = [int(value) for value in conv_channels]
        if len(channels) != 3:
            raise ValueError("ConvGRUSequenceEncoder requires exactly three conv_channels values.")
        blocks = []
        in_channels = feature_dim
        for out_channels in channels:
            blocks.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=int(kernel_size), padding=int(kernel_size) // 2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout(float(dropout)),
                ]
            )
            in_channels = out_channels
        self.conv = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            input_size=channels[-1],
            hidden_size=int(hidden_dim),
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x, lengths=None):
        x = x.float()
        if x.ndim != 3:
            raise ValueError(f"ConvGRUSequenceEncoder expects [batch, time, feature], got {tuple(x.shape)}")
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        pooled_lengths = None
        if lengths is not None:
            pooled_lengths = torch.div(lengths.to(dtype=torch.long), 8, rounding_mode="floor").clamp_min(1)
            pooled_lengths = pooled_lengths.clamp_max(x.shape[1])
        output, _ = self.gru(x)
        return _masked_sequence_mean(output, pooled_lengths)


ENCODER_REGISTRY = {
    "time_series": TimeSeriesEncoder,
    "timeseries": TimeSeriesEncoder,
    "cnn1d": TimeSeriesEncoder,
    "cnn_gru": TimeSeriesEncoder,
    "image": ImageEncoder,
    "resnet18": ImageEncoder,
    "video": VideoEncoder,
    "audio": AudioEncoder,
    "mlp": MLPEncoder,
    "gru": GRUSequenceEncoder,
    "conv_gru": ConvGRUSequenceEncoder,
}


def _encoder_cfg(cfg):
    return cfg.get("model", {}).get("encoder", {})


def resolve_encoder_type(cfg, payload_encoder_type=None):
    if payload_encoder_type:
        return str(payload_encoder_type).lower()
    enc_cfg = _encoder_cfg(cfg)
    return str(enc_cfg.get("type", "time_series")).lower()


def create_client_encoder(cfg, input_shape=None, encoder_type=None, input_dim=None):
    hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
    enc_cfg = _encoder_cfg(cfg)
    key = resolve_encoder_type(cfg, payload_encoder_type=encoder_type)
    if key not in ENCODER_REGISTRY:
        raise ValueError(f"Unsupported encoder type: {key}. Available: {sorted(ENCODER_REGISTRY)}")
    if input_shape is None and input_dim is not None:
        input_shape = [int(input_dim)]
    if key == "mlp":
        dim = int(input_dim or flattened_dim(input_shape))
        return MLPEncoder(dim, hidden_dim)
    if key == "gru":
        return GRUSequenceEncoder(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            num_layers=int(enc_cfg.get("gru_layers", 1)),
            dropout=float(enc_cfg.get("dropout", 0.1)),
            bidirectional=bool(enc_cfg.get("bidirectional", False)),
        )
    if key == "conv_gru":
        return ConvGRUSequenceEncoder(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            conv_channels=enc_cfg.get("conv_channels", [32, 64, 128]),
            kernel_size=int(enc_cfg.get("kernel_size", 5)),
            dropout=float(enc_cfg.get("dropout", 0.1)),
        )
    if key in {"time_series", "timeseries", "cnn1d", "cnn_gru"}:
        return TimeSeriesEncoder(
            input_shape=input_shape,
            hidden_dim=hidden_dim,
            conv_channels=enc_cfg.get("conv_channels", [64, 128]),
            kernel_sizes=enc_cfg.get("kernel_sizes", [5, 3]),
            dropout=float(enc_cfg.get("dropout", 0.0)),
        )
    if key in {"image", "resnet18"}:
        return ImageEncoder(input_shape, hidden_dim, pretrained=bool(enc_cfg.get("pretrained", False)))
    if key == "video":
        return VideoEncoder(input_shape, hidden_dim)
    if key == "audio":
        return AudioEncoder(input_shape, hidden_dim)
    raise ValueError(f"Unsupported encoder type: {key}")


def flattened_dim(input_shape):
    return int(math.prod([int(v) for v in input_shape]))


CNNGRUEncoder = TimeSeriesEncoder


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

    def adapt_slots(self, slot_activations: dict[int, torch.Tensor]):
        missing = [cluster_id for cluster_id in self.cluster_ids if cluster_id not in slot_activations]
        if missing:
            raise ValueError(f"Missing fusion slot activations for pred_cluster ids: {missing}")

        adapted = {}
        batch_size = None
        for slot in range(len(self.cluster_ids)):
            cluster_id = self.slot_to_cluster[slot]
            activation = slot_activations[cluster_id]
            if batch_size is None:
                batch_size = int(activation.shape[0])
            elif int(activation.shape[0]) != batch_size:
                raise ValueError("All fusion slot activations must share the same batch size.")
            adapted[cluster_id] = self.adapters[str(cluster_id)](activation)
        return adapted

    def classify_adapted(self, adapted_slots: dict[int, torch.Tensor]):
        missing = [cluster_id for cluster_id in self.cluster_ids if cluster_id not in adapted_slots]
        if missing:
            raise ValueError(f"Missing adapted fusion slots for pred_cluster ids: {missing}")
        ordered = [
            adapted_slots[self.slot_to_cluster[slot]]
            for slot in range(len(self.cluster_ids))
        ]
        fused = torch.cat(ordered, dim=1)
        return self.classifier(fused), fused

    def forward(self, slot_activations: dict[int, torch.Tensor]):
        return self.classify_adapted(self.adapt_slots(slot_activations))


def load_fusion_server_state_dict_compatible(server: ConcatMLPFusionServer, state_dict: dict):
    missing, unexpected = server.load_state_dict(state_dict, strict=False)
    unsupported_missing = [key for key in missing if not key.startswith("adapters.")]
    if unsupported_missing or unexpected:
        raise RuntimeError(
            "Incompatible fusion server checkpoint. "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    return {"missing": list(missing), "unexpected": list(unexpected)}
