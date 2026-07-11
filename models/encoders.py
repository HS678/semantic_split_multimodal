import math

import torch
from torch import nn


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


class CNNGRUEncoder(nn.Module):
    def __init__(
        self,
        input_shape,
        hidden_dim,
        conv_channels=(64, 128),
        kernel_sizes=(5, 3),
        dropout=0.0,
    ):
        super().__init__()
        if input_shape is None or len(input_shape) != 2:
            raise ValueError(f"CNNGRUEncoder expects input_shape=[channels, length], got {input_shape}")
        in_channels = int(input_shape[0])
        self.input_shape = [int(input_shape[0]), int(input_shape[1])]
        c1, c2 = [int(c) for c in conv_channels]
        k1, k2 = [int(k) for k in kernel_sizes]
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.gru = nn.GRU(input_size=c2, hidden_size=int(hidden_dim), batch_first=True)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x):
        if x.dim() == 2:
            channels, length = self.input_shape
            expected = channels * length
            if int(x.shape[1]) != expected:
                raise ValueError(f"Cannot reshape flattened input dim {int(x.shape[1])} to {self.input_shape}")
            x = x.reshape(x.shape[0], channels, length)
        z = self.conv(x)
        z = z.transpose(1, 2)
        _, h = self.gru(z)
        return self.dropout(h[-1])


def _encoder_cfg(cfg):
    return cfg.get("model", {}).get("encoder", {})


def resolve_encoder_type(cfg, modality_name=None):
    enc_cfg = _encoder_cfg(cfg)
    by_modality = enc_cfg.get("by_modality", {})
    if modality_name is not None and modality_name in by_modality:
        return str(by_modality[modality_name]).lower()
    return str(enc_cfg.get("type", "mlp")).lower()


def create_client_encoder(cfg, input_dim, input_shape=None, modality_name=None):
    hidden_dim = int(cfg.get("encoder_hidden_dim", 128))
    encoder_type = resolve_encoder_type(cfg, modality_name)
    if encoder_type == "mlp":
        return MLPEncoder(input_dim, hidden_dim)
    if encoder_type == "cnn_gru":
        enc_cfg = _encoder_cfg(cfg)
        conv_channels = enc_cfg.get("conv_channels", [64, 128])
        kernel_sizes = enc_cfg.get("kernel_sizes", [5, 3])
        dropout = float(enc_cfg.get("dropout", 0.0))
        return CNNGRUEncoder(input_shape, hidden_dim, conv_channels, kernel_sizes, dropout)
    raise ValueError(f"Unsupported encoder type: {encoder_type}. Expected 'mlp' or 'cnn_gru'.")


def flattened_dim(input_shape):
    return int(math.prod([int(v) for v in input_shape]))
