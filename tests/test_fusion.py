import torch

from semantic_split_multimodal.learning.models import (
    ConcatMLPFusionServer,
    load_fusion_server_state_dict_compatible,
)


def test_concat_mlp_fusion_uses_pred_cluster_slots_in_sorted_order():
    cfg = {
        "fusion": {"adapter_dim": 3, "hidden_dim": 5, "num_layers": 1, "dropout": 0.0},
        "model": {"server": {}},
    }
    server = ConcatMLPFusionServer(cluster_ids=[2, 0], feature_dim=4, num_classes=6, cfg=cfg)

    logits, fused = server(
        {
            0: torch.randn(7, 4),
            2: torch.randn(7, 4),
        }
    )

    assert server.cluster_ids == [0, 2]
    assert logits.shape == (7, 6)
    assert fused.shape == (7, 6)


def test_concat_mlp_fusion_accepts_explicit_noncontiguous_cluster_to_slot_mapping():
    cfg = {"fusion": {"adapter_dim": 3, "hidden_dim": 5}, "model": {"server": {}}}
    server = ConcatMLPFusionServer(
        cluster_ids=[10, 30],
        feature_dim=4,
        num_classes=6,
        cfg=cfg,
        cluster_to_slot={30: 0, 10: 1},
    )

    logits, fused = server({10: torch.randn(2, 4), 30: torch.randn(2, 4)})

    assert server.cluster_to_slot == {30: 0, 10: 1}
    assert logits.shape == (2, 6)
    assert fused.shape == (2, 6)


def test_concat_mlp_fusion_exposes_adapted_slots_for_mmbind_training():
    cfg = {"fusion": {"adapter_dim": 3, "hidden_dim": 5}, "model": {"server": {}}}
    server = ConcatMLPFusionServer(cluster_ids=[0, 1], feature_dim=4, num_classes=6, cfg=cfg)
    activations = {0: torch.randn(2, 4), 1: torch.randn(2, 4)}

    adapted = server.adapt_slots(activations)
    logits, fused = server.classify_adapted(adapted)

    assert set(adapted) == {0, 1}
    assert all(value.shape == (2, 3) for value in adapted.values())
    assert logits.shape == (2, 6)
    assert fused.shape == (2, 6)


def test_concat_mlp_fusion_rejects_incomplete_cluster_slots():
    cfg = {"fusion": {"adapter_dim": 3, "hidden_dim": 5}, "model": {"server": {}}}
    server = ConcatMLPFusionServer(cluster_ids=[0, 1], feature_dim=4, num_classes=6, cfg=cfg)

    try:
        server({0: torch.randn(2, 4)})
    except ValueError as exc:
        assert "Missing fusion slot" in str(exc)
    else:
        raise AssertionError("Expected incomplete fusion slots to raise ValueError.")


def test_concat_mlp_fusion_compatible_loads_classifier_only_legacy_checkpoint():
    cfg = {"fusion": {"adapter_dim": 3, "hidden_dim": 5}, "model": {"server": {}}}
    server = ConcatMLPFusionServer(cluster_ids=[0, 1], feature_dim=4, num_classes=6, cfg=cfg)
    legacy_state = {
        key: value.clone()
        for key, value in server.state_dict().items()
        if key.startswith("classifier.")
    }

    result = load_fusion_server_state_dict_compatible(server, legacy_state)

    assert result["unexpected"] == []
    assert all(key.startswith("adapters.") for key in result["missing"])
