from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from semantic_split_multimodal.evaluation.metrics import learning_metrics
from semantic_split_multimodal.evaluation.oracle_mapping import SUCCESS


def evaluate_naturally_paired_fusion(
    server,
    clients_by_id,
    test_multimodal_path: Path,
    oracle_mapping: dict,
    cfg: dict,
    device,
):
    if oracle_mapping.get("status") != SUCCESS:
        return {
            "eval_status": "failed",
            "eval_failure_reason": oracle_mapping.get("failure_reason", "evaluation_mapping_failure"),
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
        }

    test_payload = torch.load(Path(test_multimodal_path), map_location="cpu")
    labels = test_payload["label"].long()
    modality_names = list(test_payload["modality_names"])
    modalities = test_payload["modalities"]
    modality_to_cluster = {int(k): int(v) for k, v in oracle_mapping["modality_to_cluster"].items()}
    representative_clients = {int(k): v for k, v in oracle_mapping["representative_clients"].items()}

    missing_modalities = [idx for idx in range(len(modality_names)) if idx not in modality_to_cluster]
    if missing_modalities:
        return {
            "eval_status": "failed",
            "eval_failure_reason": "evaluation_mapping_incomplete",
            "loss": None,
            "accuracy": None,
            "macro_f1": None,
        }

    tensors = [modalities[name] for name in modality_names]
    dataset = TensorDataset(*tensors, labels)
    batch_size = int(cfg.get("training", {}).get("eval_batch_size", cfg.get("training", {}).get("batch_size", 64)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ce = nn.CrossEntropyLoss(reduction="sum")
    y_true, y_pred = [], []
    total_loss = 0.0
    total = 0
    num_eval_batches = 0
    eval_modality_ids = list(range(len(tensors)))
    eval_cluster_ids = sorted(modality_to_cluster.values())

    server.eval()
    used_clients = []
    for client_id in representative_clients.values():
        client = clients_by_id[client_id]
        client.encoder.eval()
        used_clients.append(client)

    with torch.no_grad():
        for batch in loader:
            *xs, yb = batch
            slot_activations = {}
            for modality_id, xb in enumerate(xs):
                client_id = representative_clients[modality_id]
                cluster_id = modality_to_cluster[modality_id]
                client = clients_by_id[client_id]
                slot_activations[cluster_id] = client.encoder(xb.to(device))
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
            "num_eval_samples": int(total),
            "num_eval_batches": int(num_eval_batches),
            "eval_modality_ids": eval_modality_ids,
            "eval_cluster_ids": eval_cluster_ids,
            "oracle_mapping_type": oracle_mapping.get("mapping_type", "oracle_evaluation_only"),
        }
    )
    return metrics
