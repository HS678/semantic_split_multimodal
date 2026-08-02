import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def encoder_fingerprint(encoder, samples, lengths=None, batch_size=64, max_batches=4, device="cpu"):
    dataset = TensorDataset(samples, lengths) if lengths is not None else TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    chunks = []
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            xb = batch[0].to(device)
            length_batch = batch[1].to(device) if len(batch) > 1 else None
            encoded = encoder(xb) if length_batch is None else encoder(xb, length_batch)
            chunks.append(encoded.detach().cpu())
    if not chunks:
        raise RuntimeError("Cannot extract fingerprint from an empty client dataset.")
    z = torch.cat(chunks, dim=0)
    return torch.cat([z.mean(dim=0), z.std(dim=0)], dim=0).numpy().astype(np.float32)


def signal_stat_fingerprint(samples):
    x = samples.detach().float()
    if x.dim() == 1:
        x = x.reshape(-1, 1, 1)
    elif x.dim() == 2:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    reduce_dims = tuple(i for i in range(x.dim()) if i != 1)
    stats = [
        x.mean(dim=reduce_dims),
        x.std(dim=reduce_dims),
        x.abs().mean(dim=reduce_dims),
        x.amax(dim=reduce_dims),
        x.amin(dim=reduce_dims),
    ]
    return torch.cat(stats).numpy().astype(np.float32)


def pad_feature_rows(rows):
    max_len = max(len(row) for row in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def build_fingerprints(clients, encoders, cfg, device):
    fp_cfg = cfg.get("fingerprint", {})
    source = str(fp_cfg.get("type", fp_cfg.get("source", "hybrid"))).lower()
    batch_size = int(fp_cfg.get("batch_size", cfg.get("batch_size", 64)))
    max_batches = fp_cfg.get("max_batches", 4)
    rows = []
    for client in clients:
        parts = []
        if source in {"encoder", "hybrid"}:
            parts.append(
                encoder_fingerprint(
                    encoders[client.client_id],
                    client.samples,
                    client.sequence_lengths,
                    batch_size,
                    max_batches,
                    device,
                )
            )
        if source in {"signal", "signal_stats", "hybrid"}:
            parts.append(signal_stat_fingerprint(client.samples))
        if not parts:
            raise ValueError("fingerprint.type must be 'encoder', 'signal', or 'hybrid'.")
        rows.append(np.concatenate(parts).astype(np.float32))
    return pad_feature_rows(rows)
