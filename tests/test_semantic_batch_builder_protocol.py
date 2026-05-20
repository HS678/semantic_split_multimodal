import torch

from server.server_core import SemanticBatchBuilder


def _payload(client_id, modality_id, labels, dim=4):
    b = len(labels)
    z = torch.randn(b, dim, requires_grad=True)
    return {
        "client_id": client_id,
        "modality_id": modality_id,
        "y": torch.tensor(labels, dtype=torch.long),
        "z_server": z,
    }


def test_semantic_batch_builder_uses_common_labels_only():
    p1 = _payload("c1", 0, [0, 0, 1, 2, 2])
    p2 = _payload("c2", 1, [1, 1, 2, 2, 3])

    out = SemanticBatchBuilder.build([p1, p2])
    assert out is not None

    # Common labels are {1, 2}; label 1 count=min(1,2)=1, label 2 count=min(2,2)=2 => total 3
    assert out["common_labels"] == [1, 2]
    assert out["semantic_labels"].tolist() == [1, 2, 2]

    idx1 = out["index_map"]["c1"]
    idx2 = out["index_map"]["c2"]
    assert len(idx1) == 3 and len(idx2) == 3

    y1 = p1["y"][torch.tensor(idx1)]
    y2 = p2["y"][torch.tensor(idx2)]
    assert torch.equal(y1, out["semantic_labels"])
    assert torch.equal(y2, out["semantic_labels"])


def test_semantic_batch_builder_no_common_label_returns_none():
    p1 = _payload("c1", 0, [0, 0, 0])
    p2 = _payload("c2", 1, [1, 1, 1])
    out = SemanticBatchBuilder.build([p1, p2])
    assert out is None
