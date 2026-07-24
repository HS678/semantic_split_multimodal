import torch

from models.fusion import ConcatMLPFusionServer


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


def test_concat_mlp_fusion_rejects_incomplete_cluster_slots():
    cfg = {"fusion": {"adapter_dim": 3, "hidden_dim": 5}, "model": {"server": {}}}
    server = ConcatMLPFusionServer(cluster_ids=[0, 1], feature_dim=4, num_classes=6, cfg=cfg)

    try:
        server({0: torch.randn(2, 4)})
    except ValueError as exc:
        assert "Missing fusion slot" in str(exc)
    else:
        raise AssertionError("Expected incomplete fusion slots to raise ValueError.")
