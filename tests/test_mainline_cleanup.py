import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from semantic_split_multimodal.learning.fusion_sl import _save_checkpoint
from semantic_split_multimodal.learning.models import ConcatMLPFusionServer
from semantic_split_multimodal.learning.scheduling import build_scheduler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tracked_runtime_code_has_no_legacy_method_imports():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=PROJECT_ROOT, text=True).splitlines()
    runtime_files = [
        path
        for path in tracked
        if path.startswith(("src/", "scripts/", "configs/", "tests/"))
        and path.endswith((".py", ".config"))
    ]

    offenders = []
    for rel_path in runtime_files:
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        legacy_tokens = [
            "baseline" + "_unpaired",
            "baseline" + "_eval",
            "unpaired" + "_split_learning",
        ]
        if any(token in text for token in legacy_tokens):
            offenders.append(rel_path)

    assert offenders == []


class SchedulerClient:
    def __init__(self, client_id, pred_cluster):
        self.client_id = client_id
        self.pred_cluster = pred_cluster

    @property
    def hidden_modality_id(self):
        raise AssertionError("Mainline scheduler must not read hidden_modality_id.")


def test_balanced_scheduler_samples_equal_clients_per_cluster_without_hidden_modality_id():
    clients = [
        *(SchedulerClient(f"c0_{idx}", 0) for idx in range(8)),
        *(SchedulerClient(f"c1_{idx}", 1) for idx in range(8)),
    ]
    scheduler = build_scheduler("balanced_cluster_round_robin", clients, clients_per_cluster_per_round=3, seed=7)

    rounds = [scheduler.sample_round() for _ in range(3)]

    for selected in rounds:
        counts = {0: 0, 1: 0}
        for client in selected:
            counts[int(client.pred_cluster)] += 1
        assert counts == {0: 3, 1: 3}
        assert len({client.client_id for client in selected}) == 6

    cluster0_seen_before_round3 = {
        client.client_id
        for selected in rounds[:2]
        for client in selected
        if int(client.pred_cluster) == 0
    }
    cluster0_round3 = [
        client.client_id
        for client in rounds[2]
        if int(client.pred_cluster) == 0
    ]
    assert len(cluster0_seen_before_round3) == 6
    assert sum(client_id not in cluster0_seen_before_round3 for client_id in cluster0_round3) == 2
    assert sum(client_id in cluster0_seen_before_round3 for client_id in cluster0_round3) == 1


def test_balanced_scheduler_rejects_unsupported_names_and_oversized_r():
    clients = [SchedulerClient("c0", 0), SchedulerClient("c1", 0), SchedulerClient("c2", 1), SchedulerClient("c3", 1)]

    with pytest.raises(ValueError, match="balanced_cluster_round_robin"):
        build_scheduler("unsupported_scheduler", clients, clients_per_cluster_per_round=1, seed=7)

    with pytest.raises(ValueError, match="cannot exceed"):
        build_scheduler("balanced_cluster_round_robin", clients, clients_per_cluster_per_round=3, seed=7)


class CheckpointClient:
    def __init__(self, client_id, pred_cluster, module):
        self.client_id = client_id
        self.pred_cluster = pred_cluster
        self.encoder = module
        self.device = torch.device("cpu")


def test_fusion_checkpoint_saves_and_reloads_server_and_client_encoders(tmp_path):
    cfg = {
        "encoder_hidden_dim": 4,
        "num_classes": 2,
        "fusion": {"adapter_dim": 3, "hidden_dim": 5, "num_layers": 1, "dropout": 0.0},
        "model": {"server": {}},
    }
    server = ConcatMLPFusionServer([0, 1], feature_dim=4, num_classes=2, cfg=cfg)
    clients = [
        CheckpointClient("client_000", 0, nn.Linear(2, 4)),
        CheckpointClient("client_010", 1, nn.Linear(2, 4)),
    ]
    path = tmp_path / "best_model.pt"

    _save_checkpoint(
        path,
        server,
        clients,
        cfg,
        cluster_ids=[0, 1],
        cluster_to_slot={0: 0, 1: 1},
        metrics={"eval_status": "success", "accuracy": 1.0},
    )

    payload = torch.load(path, map_location="cpu")
    reloaded = ConcatMLPFusionServer(payload["cluster_ids"], 4, 2, cfg, payload["cluster_to_slot"])
    reloaded.load_state_dict(payload["server_state_dict"])

    for name, value in server.state_dict().items():
        assert torch.equal(value, reloaded.state_dict()[name])
    assert payload["pred_cluster_assignments"] == {"client_000": 0, "client_010": 1}
    assert set(payload["client_encoder_states"]) == {"client_000", "client_010"}
