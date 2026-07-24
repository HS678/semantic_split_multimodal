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

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
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

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))


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
