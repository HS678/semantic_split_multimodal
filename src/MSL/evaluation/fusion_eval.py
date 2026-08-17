from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from MSL.evaluation.metrics import learning_metrics
from MSL.evaluation.oracle_mapping import SUCCESS
from MSL.evaluation.routing import build_tolerant_evaluation_routing, route_paired_batch


# 使用 naturally paired test 数据执行 tolerant fusion evaluation。
def evaluate_naturally_paired_fusion(
    server,
    clients_by_id,
    multimodal_path: Path,
    oracle_mapping: dict,
    cfg: dict,
    device,
):
    if oracle_mapping.get("status") != SUCCESS:
        return {
            "eval_status": "failed",
            "eval_failure_reason": oracle_mapping.get("failure_reason", "evaluation_mapping_failure"),
            "loss": None,
            "classification_loss": None,
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
        }

    payload = torch.load(Path(multimodal_path), map_location="cpu")
    labels = payload["label"].long()
    modality_names = list(payload["modality_names"])
    modalities = payload["modalities"]
    modality_lengths = payload.get("modality_lengths")
    routing = build_tolerant_evaluation_routing(
        Path(cfg["evaluation"]["client_meta_path"]),
        Path(cfg["evaluation"]["cluster_assignment_path"]),
        clients_by_id,
        cfg["evaluation"].get("cluster_assignment_column", "pred_cluster"),
    )

    tensors = [modalities[name] for name in modality_names]
    length_tensors = None
    if modality_lengths is not None:
        length_tensors = [modality_lengths[name] for name in modality_names]
        dataset = TensorDataset(*tensors, *length_tensors, labels)
    else:
        dataset = TensorDataset(*tensors, labels)
    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ce = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred = [], []
    total_loss = 0.0
    total = 0
    num_eval_batches = 0
    eval_modality_ids = list(range(len(tensors)))
    eval_cluster_ids = routing.pred_clusters

    server.eval()
    used_clients = list(clients_by_id.values())
    for client in used_clients:
        client.encoder.eval()

    with torch.no_grad():
        for batch in loader:
            if length_tensors is None:
                xs = list(batch[:-1])
                batch_lengths = [None] * len(xs)
            else:
                xs = list(batch[: len(tensors)])
                batch_lengths = list(batch[len(tensors) : 2 * len(tensors)])
            yb = batch[-1]
            slot_activations = route_paired_batch(xs, batch_lengths, routing, device)
            logits, _ = server(slot_activations)
            pred = logits.argmax(dim=1)
            y_true.extend(yb.tolist())
            y_pred.extend(pred.detach().cpu().tolist())
            total_loss += float(ce(logits, yb.to(device)).item())
            total += int(yb.numel())
            num_eval_batches += 1

    server.train()
    for client in used_clients:
        client.encoder.train()

    metrics = learning_metrics(y_true, y_pred)
    metrics.update(
        {
            "eval_status": "success",
            "eval_failure_reason": None,
            "loss": float(total_loss / max(1, total)),
            "classification_loss": float(total_loss / max(1, total)),
            "loss_type": "classification_cross_entropy",
            "num_eval_samples": int(total),
            "num_eval_batches": int(num_eval_batches),
            "eval_modality_ids": eval_modality_ids,
            "eval_cluster_ids": eval_cluster_ids,
            "oracle_mapping_type": oracle_mapping.get("mapping_type", "tolerant_evaluation_only"),
            "tolerant_routing": routing.to_metadata(),
        }
    )
    return metrics
