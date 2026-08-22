import csv
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from MSL.binding import build_label_random_pseudo_batch
from MSL.evaluation import (
    build_physical_evaluation_routing,
    evaluate_naturally_paired_fusion,
    route_physical_paired_batch,
)
from MSL.models import ConcatMLPFusionServer
from MSL.protocol import TRAINING_METHODS
from MSL.scheduling import BalancedClusterRoundRobinScheduler, RandomScheduler
from MSL.training import _collect_selected_activations


@dataclass
class SlotClient:
    client_id: str
    pred_cluster: int
    hidden_modality_id: int

    def sample_batch(self):
        base = float(int(self.hidden_modality_id) + 1)
        return torch.full((3, 2), base), torch.tensor([0, 1, 0]), None

    def forward_detached(self, x, lengths=None):
        z_client = x.clone().requires_grad_(True)
        z_server = z_client.detach().requires_grad_(True)
        return z_client, z_server


class TinyEncoder(nn.Module):
    def __init__(self, multiplier: float):
        super().__init__()
        self.multiplier = float(multiplier)

    def forward(self, x, lengths=None):
        return x * self.multiplier


class EvalClient:
    def __init__(self, client_id: str, multiplier: float):
        self.client_id = client_id
        self.encoder = TinyEncoder(multiplier)


def _write_client_meta(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "client_id",
                "hidden_modality_id",
                "hidden_modality_name",
                "num_samples",
                "encoder_type",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "client_id": row["client_id"],
                    "hidden_modality_id": row["hidden_modality_id"],
                    "hidden_modality_name": f"m{row['hidden_modality_id']}",
                    "num_samples": 1,
                    "encoder_type": "tiny",
                }
            )


def test_pred_cluster_affects_scheduler_selection_only():
    clients = [
        SlotClient("a0", pred_cluster=0, hidden_modality_id=0),
        SlotClient("a1", pred_cluster=1, hidden_modality_id=0),
        SlotClient("b0", pred_cluster=1, hidden_modality_id=1),
        SlotClient("b1", pred_cluster=1, hidden_modality_id=1),
    ]
    scheduler = BalancedClusterRoundRobinScheduler(clients, clients_per_round=2, seed=0)
    selected = scheduler.sample_round()
    assert len(selected) == 2
    assert scheduler.metrics(selected)["clients_per_round"] == 2

    batches, _ = _collect_selected_activations(selected)
    assert {batch.pred_cluster for batch in batches} == {
        client.hidden_modality_id for client in selected
    }


def test_server_slot_count_is_physical_q_when_predicted_q_differs():
    server = ConcatMLPFusionServer(
        [0, 1, 2],
        2,
        2,
        {"fusion": {"adapter_dim": 2, "hidden_dim": 4}},
    )
    assert server.cluster_ids == [0, 1, 2]
    assert len(server.adapters) == 3


def test_bad_merge_routes_a_b_clients_to_separate_physical_slots():
    selected = [
        SlotClient("a0", pred_cluster=0, hidden_modality_id=0),
        SlotClient("b0", pred_cluster=0, hidden_modality_id=1),
        SlotClient("c0", pred_cluster=1, hidden_modality_id=2),
    ]
    batches, _ = _collect_selected_activations(selected)
    pseudo = build_label_random_pseudo_batch(
        batches,
        required_clusters=[0, 1, 2],
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
        allow_missing_clusters_with_zero=True,
    )
    assert pseudo is not None
    assert sorted(pseudo.slot_activations) == [0, 1, 2]
    assert {batch.pred_cluster for batch in batches} == {0, 1, 2}


def test_bad_split_maps_one_physical_modality_to_one_fusion_slot():
    selected = [
        SlotClient("a0", pred_cluster=0, hidden_modality_id=0),
        SlotClient("a1", pred_cluster=1, hidden_modality_id=0),
    ]
    batches, _ = _collect_selected_activations(selected)
    assert [batch.pred_cluster for batch in batches] == [0, 0]


def test_missing_physical_modality_is_zero_filled():
    selected = [SlotClient("a0", pred_cluster=0, hidden_modality_id=0)]
    batches, _ = _collect_selected_activations(selected)
    pseudo = build_label_random_pseudo_batch(
        batches,
        required_clusters=[0, 1],
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
        allow_missing_clusters_with_zero=True,
    )
    assert pseudo is not None
    assert torch.equal(pseudo.slot_activations[1], torch.zeros_like(pseudo.slot_activations[1]))
    assert pseudo.source_client_ids[1] == []


def test_randomsl_uses_same_physical_slot_construction_after_selection():
    clients = [
        SlotClient("a0", pred_cluster=99, hidden_modality_id=0),
        SlotClient("b0", pred_cluster=42, hidden_modality_id=1),
        SlotClient("b1", pred_cluster=42, hidden_modality_id=1),
    ]
    selected = RandomScheduler(clients, clients_per_round=2, seed=3).sample_round()
    batches, _ = _collect_selected_activations(selected)
    assert {batch.pred_cluster for batch in batches} == {
        client.hidden_modality_id for client in selected
    }


def test_evaluation_routing_does_not_depend_on_test_labels(tmp_path):
    meta_path = tmp_path / "client_meta.csv"
    rows = [
        {"client_id": "m0_a", "hidden_modality_id": 0},
        {"client_id": "m1_a", "hidden_modality_id": 1},
    ]
    _write_client_meta(meta_path, rows)
    clients = {"m0_a": EvalClient("m0_a", 2.0), "m1_a": EvalClient("m1_a", 3.0)}
    routing_a = build_physical_evaluation_routing(meta_path, clients).to_metadata()
    routing_b = build_physical_evaluation_routing(meta_path, clients).to_metadata()
    assert routing_a["uses_test_labels_for_routing"] is False
    assert routing_a["canonical_client_ids_by_modality"] == routing_b["canonical_client_ids_by_modality"]


def test_evaluator_routes_naturally_paired_physical_modalities_directly(tmp_path):
    meta_path = tmp_path / "client_meta.csv"
    rows = [
        {"client_id": "m0_a", "hidden_modality_id": 0},
        {"client_id": "m1_a", "hidden_modality_id": 1},
    ]
    _write_client_meta(meta_path, rows)
    clients = {"m0_a": EvalClient("m0_a", 2.0), "m1_a": EvalClient("m1_a", 3.0)}
    routing = build_physical_evaluation_routing(meta_path, clients)
    slots = route_physical_paired_batch(
        [torch.ones(2, 2), torch.ones(2, 2) * 5],
        [None, None],
        routing,
        torch.device("cpu"),
    )
    assert sorted(slots) == [0, 1]
    assert torch.equal(slots[0], torch.ones(2, 2) * 2)
    assert torch.equal(slots[1], torch.ones(2, 2) * 15)


def test_new_formal_evaluator_has_no_p_mq_weighted_mixing(tmp_path):
    meta_path = tmp_path / "client_meta.csv"
    _write_client_meta(meta_path, [{"client_id": "m0_a", "hidden_modality_id": 0}])
    routing = build_physical_evaluation_routing(meta_path, {"m0_a": EvalClient("m0_a", 1.0)})
    metadata = routing.to_metadata()
    assert metadata["uses_p_mq_weighted_mixing"] is False
    assert "P_mq" not in metadata


def test_formal_evaluator_entry_uses_physical_slots_without_cluster_assignment(tmp_path):
    meta_path = tmp_path / "client_meta.csv"
    _write_client_meta(
        meta_path,
        [
            {"client_id": "m0_a", "hidden_modality_id": 0},
            {"client_id": "m1_a", "hidden_modality_id": 1},
        ],
    )
    multimodal_path = tmp_path / "test_multimodal.pt"
    torch.save(
        {
            "label": torch.tensor([0, 1]),
            "modality_names": ["m0", "m1"],
            "modalities": {
                "m0": torch.ones(2, 2),
                "m1": torch.ones(2, 2) * 2,
            },
        },
        multimodal_path,
    )
    cfg = {
        "training": {"batch_size": 2},
        "evaluation": {"client_meta_path": str(meta_path)},
        "fusion": {"adapter_dim": 2, "hidden_dim": 4},
    }
    clients = {"m0_a": EvalClient("m0_a", 1.0), "m1_a": EvalClient("m1_a", 1.0)}
    server = ConcatMLPFusionServer([0, 1], 2, 2, cfg)

    metrics = evaluate_naturally_paired_fusion(
        server,
        clients,
        multimodal_path,
        oracle_mapping=None,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    assert metrics["eval_status"] == "success"
    assert metrics["eval_slot_ids"] == [0, 1]
    assert metrics["eval_cluster_ids"] == []
    assert metrics["tolerant_routing"] is None
    assert metrics["evaluation_mapping"]["uses_p_mq_weighted_mixing"] is False
    assert metrics["evaluation_mapping"]["uses_test_labels_for_routing"] is False


def test_evaluation_encoder_choice_is_deterministic_and_method_independent(tmp_path):
    meta_path = tmp_path / "client_meta.csv"
    rows = [
        {"client_id": "m0_b", "hidden_modality_id": 0},
        {"client_id": "m0_a", "hidden_modality_id": 0},
        {"client_id": "m1_a", "hidden_modality_id": 1},
    ]
    _write_client_meta(meta_path, rows)
    clients = {
        "m0_b": EvalClient("m0_b", 9.0),
        "m0_a": EvalClient("m0_a", 2.0),
        "m1_a": EvalClient("m1_a", 3.0),
    }
    metadata = build_physical_evaluation_routing(meta_path, clients).to_metadata()
    assert metadata["evaluation_encoder_rule"] == "canonical_client_per_physical_modality_by_sorted_client_id"
    assert metadata["canonical_client_ids_by_modality"] == {"0": "m0_a", "1": "m1_a"}
    assert metadata["uses_pred_cluster_for_routing"] is False


def test_equivalent_partitions_up_to_label_permutation_have_equivalent_scheduler_behavior():
    original = [
        SlotClient("a0", pred_cluster=0, hidden_modality_id=0),
        SlotClient("a1", pred_cluster=0, hidden_modality_id=0),
        SlotClient("b0", pred_cluster=1, hidden_modality_id=1),
        SlotClient("b1", pred_cluster=1, hidden_modality_id=1),
    ]
    permuted = [
        SlotClient(client.client_id, pred_cluster=1 - client.pred_cluster, hidden_modality_id=client.hidden_modality_id)
        for client in original
    ]
    first = BalancedClusterRoundRobinScheduler(original, clients_per_round=2, seed=11).sample_round()
    second = BalancedClusterRoundRobinScheduler(permuted, clients_per_round=2, seed=11).sample_round()
    assert sorted(client.client_id for client in first) == sorted(client.client_id for client in second)


def test_formal_c2_method_list_excludes_oracle():
    assert TRAINING_METHODS == (
        "randomsl",
        "kmeans2",
        "kmeans3",
        "kmeans4",
        "kmeans5",
        "auto_kmeans",
        "gmm_bic",
        "ours",
    )
    assert "oracle" not in TRAINING_METHODS
