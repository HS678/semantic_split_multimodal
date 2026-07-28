import subprocess
from pathlib import Path

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
        and path.endswith((".py", ".yaml", ".yml"))
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


def test_proposed_scheduler_covers_pred_clusters_without_hidden_modality_id():
    clients = [
        SchedulerClient("c0", 0),
        SchedulerClient("c1", 0),
        SchedulerClient("c2", 1),
        SchedulerClient("c3", 1),
        SchedulerClient("c4", 2),
        SchedulerClient("c5", 2),
    ]
    scheduler = build_scheduler("proposed_cluster_coverage", clients, clients_per_round=3, seed=7)

    selected = scheduler.sample_round()

    assert sorted({client.pred_cluster for client in selected}) == [0, 1, 2]


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
    path = tmp_path / "best_mmbind_fusion_checkpoint.pt"

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
