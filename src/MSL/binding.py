# 语义 binding 与 pseudo-label 置信度计算逻辑。
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class ClientActivationBatch:
    client_id: str
    pred_cluster: int
    activations: torch.Tensor
    labels: torch.Tensor


@dataclass
class PseudoMultimodalBatch:
    slot_activations: dict[int, torch.Tensor]
    labels: torch.Tensor
    binding_confidences: torch.Tensor
    source_client_ids: dict[int, list[str]]
    source_indices: dict[int, torch.Tensor]

    @property
    def batch_size(self) -> int:
        return int(self.labels.shape[0])

    @property
    def cluster_ids(self) -> list[int]:
        return sorted(self.slot_activations)


def build_label_random_pseudo_batch(
    batches: Sequence[ClientActivationBatch],
    required_clusters: Sequence[int],
    batch_size: int,
    generator: torch.Generator | None = None,
    allow_missing_clusters_with_zero: bool = False,
) -> PseudoMultimodalBatch | None:
    """Build anchor-based same-label pseudo multimodal samples.

    Each pseudo sample contains exactly one activation from every required
    predicted cluster slot. All selected non-empty activations in a pseudo
    sample share the same class label. Missing required clusters can be
    represented by zero activations when explicitly requested by the trainer.
    """

    cluster_ids = sorted({int(cluster_id) for cluster_id in required_clusters})
    if not cluster_ids:
        raise ValueError("required_clusters must not be empty.")
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("required_clusters must contain unique cluster ids.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive.")

    by_cluster = {cluster_id: [] for cluster_id in cluster_ids}
    for batch in batches:
        cluster_id = int(batch.pred_cluster)
        if cluster_id not in by_cluster:
            continue
        if batch.activations.shape[0] != batch.labels.shape[0]:
            raise ValueError(f"Activation/label length mismatch for client {batch.client_id}.")
        if batch.activations.shape[0] == 0:
            continue
        by_cluster[cluster_id].append(batch)

    present_cluster_ids = [cluster_id for cluster_id in cluster_ids if by_cluster[cluster_id]]
    if any(not group for group in by_cluster.values()) and not allow_missing_clusters_with_zero:
        return None
    if not present_cluster_ids:
        return None

    common_labels = common_labels_for_clusters(batches, present_cluster_ids)
    if not common_labels:
        return None

    device = batches[0].activations.device
    feature_shape = tuple(batches[0].activations.shape[1:])
    common_label_set = set(int(label) for label in common_labels)
    anchor_cluster = present_cluster_ids[0]
    anchor_candidates = [
        (batch, sample_index)
        for batch in by_cluster[anchor_cluster]
        for sample_index in torch.nonzero(
            torch.isin(batch.labels.detach().cpu().reshape(-1), torch.tensor(sorted(common_label_set))),
            as_tuple=False,
        ).reshape(-1).tolist()
    ]
    if not anchor_candidates:
        return None

    slot_activations = {cluster_id: [] for cluster_id in cluster_ids}
    source_client_ids = {cluster_id: [] for cluster_id in cluster_ids}
    source_indices = {cluster_id: [] for cluster_id in cluster_ids}
    out_labels = []

    for _ in range(int(batch_size)):
        anchor_pos = int(_randint(len(anchor_candidates), 1, device, generator).item())
        anchor_batch, anchor_index = anchor_candidates[anchor_pos]
        label = int(anchor_batch.labels.detach().cpu().reshape(-1)[anchor_index].item())
        out_labels.append(label)
        for cluster_id in cluster_ids:
            if not by_cluster[cluster_id]:
                slot_activations[cluster_id].append(
                    torch.zeros(feature_shape, dtype=anchor_batch.activations.dtype, device=device)
                )
                continue
            if cluster_id == anchor_cluster:
                candidates = [(anchor_batch, anchor_index)]
            else:
                candidates = _candidates_for_label(by_cluster[cluster_id], int(label))
            if not candidates:
                return None
            selected_pos = int(_randint(len(candidates), 1, device, generator).item())
            batch, sample_index = candidates[selected_pos]
            slot_activations[cluster_id].append(batch.activations[sample_index])
            source_client_ids[cluster_id].append(batch.client_id)
            source_indices[cluster_id].append(int(sample_index))

    stacked_activations = {
        cluster_id: torch.stack(values, dim=0)
        for cluster_id, values in slot_activations.items()
    }
    stacked_indices = {
        cluster_id: torch.tensor(values, dtype=torch.long, device=device)
        for cluster_id, values in source_indices.items()
    }
    labels = torch.tensor(out_labels, dtype=torch.long, device=device)
    # Exact-label binding is the label-shared special case of MMBind. Every
    # constructed tuple therefore has maximum label-similarity confidence.
    # Keeping confidence explicit lets alternative soft semantic binders reuse
    # the weighted contrastive objective without changing the training API.
    binding_confidences = torch.ones(int(labels.shape[0]), dtype=torch.float32, device=device)

    return PseudoMultimodalBatch(
        slot_activations=stacked_activations,
        labels=labels,
        binding_confidences=binding_confidences,
        source_client_ids=source_client_ids,
        source_indices=stacked_indices,
    )


def _candidates_for_label(batches: Sequence[ClientActivationBatch], label: int):
    candidates = []
    for batch in batches:
        indices = torch.nonzero(batch.labels.detach().cpu().reshape(-1) == int(label), as_tuple=False).reshape(-1)
        for index in indices.tolist():
            candidates.append((batch, int(index)))
    return candidates


def common_labels_for_clusters(
    batches: Sequence[ClientActivationBatch],
    required_clusters: Sequence[int],
) -> list[int]:
    cluster_ids = sorted({int(cluster_id) for cluster_id in required_clusters})
    by_cluster = {cluster_id: [] for cluster_id in cluster_ids}
    for batch in batches:
        cluster_id = int(batch.pred_cluster)
        if cluster_id in by_cluster and batch.labels.numel() > 0:
            by_cluster[cluster_id].append(batch.labels.detach().reshape(-1).cpu())

    if any(not labels for labels in by_cluster.values()):
        return []

    labels_by_cluster = {
        cluster_id: set(torch.cat(labels).tolist())
        for cluster_id, labels in by_cluster.items()
    }
    return sorted(int(label) for label in set.intersection(*(labels_by_cluster[cluster_id] for cluster_id in cluster_ids)))


def _randint(high: int, size: int, device, generator: torch.Generator | None):
    if generator is None:
        return torch.randint(0, int(high), (int(size),), device=device)
    return torch.randint(0, int(high), (int(size),), generator=generator, device=device)
