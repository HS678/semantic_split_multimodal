import torch

from binding import ClientActivationBatch, build_label_random_pseudo_batch


def test_label_random_binding_requires_complete_cluster_coverage():
    batches = [
        ClientActivationBatch("c0", 0, torch.randn(3, 4), torch.tensor([0, 1, 1])),
    ]

    pseudo = build_label_random_pseudo_batch(
        batches,
        required_clusters=[0, 1],
        batch_size=2,
        generator=torch.Generator().manual_seed(0),
    )

    assert pseudo is None


def test_label_random_binding_pairs_same_label_across_pred_cluster_slots():
    batches = [
        ClientActivationBatch(
            "c0",
            0,
            torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([0, 1, 1]),
        ),
        ClientActivationBatch(
            "c1",
            1,
            torch.tensor([[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]),
            torch.tensor([1, 0, 1]),
        ),
    ]

    pseudo = build_label_random_pseudo_batch(
        batches,
        required_clusters=[0, 1],
        batch_size=4,
        generator=torch.Generator().manual_seed(4),
    )

    assert pseudo is not None
    assert pseudo.batch_size == 4
    assert pseudo.cluster_ids == [0, 1]
    assert set(pseudo.slot_activations) == {0, 1}
    for cluster_id, activations in pseudo.slot_activations.items():
        assert activations.shape == (4, 2)
        labels = batches[cluster_id].labels[pseudo.source_indices[cluster_id]]
        assert torch.equal(labels, pseudo.labels.cpu())
