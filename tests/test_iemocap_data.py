import json
from pathlib import Path

import torch

from MSL.data.partitioner import run_stage1_partition
from MSL.data.registry import load_dataset
from MSL.learning.models import create_client_encoder


def _write_synthetic_iemocap_cache(tmp_path: Path):
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    raw_root.mkdir()
    processed_root.mkdir()
    sample_ids = [f"sample_{idx:02d}" for idx in range(10)]
    sessions = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    rows = [
        {
            "utterance_id": sample_id,
            "session_id": session_id,
            "label": idx % 4,
        }
        for idx, (sample_id, session_id) in enumerate(zip(sample_ids, sessions))
    ]
    (processed_root / "manifest.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )
    specifications = {
        "audio": (12, 4, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]),
        "video": (5, 6, [5] * 10),
        "text": (7, 8, [7, 6, 5, 4, 3, 7, 6, 5, 4, 3]),
    }
    for modality_name, (time, dim, lengths) in specifications.items():
        features = torch.randn(10, time, dim)
        for sample_idx, length in enumerate(lengths):
            features[sample_idx, length:] = 0.0
        torch.save(
            {
                "sample_ids": sample_ids,
                "features": features,
                "lengths": torch.tensor(lengths),
                "feature_extractor": {"type": f"synthetic_{modality_name}"},
            },
            processed_root / f"{modality_name}.pt",
        )
    return raw_root, processed_root


def _config(raw_root: Path, processed_root: Path):
    return {
        "seed": 42,
        "dataset": {
            "type": "iemocap",
            "variant": "full",
            "root": str(raw_root),
            "processed_root": str(processed_root),
            "feature_recipe": "mfcc_mobilevit_xs_distilbert_v1",
            "task": "emotion_4class",
            "label_protocol": "ang_hap_exc_sad_neu_v1",
            "split_protocol": "session_disjoint_123_4_5_v1",
            "train_sessions": [1, 2, 3],
            "validation_sessions": [4],
            "test_sessions": [5],
            "normalize": True,
        },
        "num_classes": 4,
        "encoder_hidden_dim": 16,
        "model": {
            "encoder": {
                "type": "gru",
                "conv_channels": [8, 12, 16],
                "kernel_size": 5,
                "dropout": 0.0,
            }
        },
        "partition": {"clients_per_modality": 2},
    }


def _write_loso_iemocap_cache(tmp_path: Path):
    raw_root = tmp_path / "loso_raw"
    processed_root = tmp_path / "loso_processed"
    raw_root.mkdir()
    processed_root.mkdir()
    rows = []
    for session_id in range(1, 6):
        for dialog_idx in range(8):
            dialog_id = f"Ses{session_id:02d}_dialog{dialog_idx:02d}"
            for label in range(4):
                rows.append(
                    {
                        "utterance_id": f"{dialog_id}_{label}",
                        "session_id": session_id,
                        "dialog_id": dialog_id,
                        "label": label,
                    }
                )
    sample_ids = [row["utterance_id"] for row in rows]
    (processed_root / "manifest.json").write_text(
        json.dumps({"samples": rows}), encoding="utf-8"
    )
    for modality_name, time, dim in (
        ("audio", 12, 4),
        ("video", 8, 6),
        ("text", 10, 8),
    ):
        torch.save(
            {
                "sample_ids": sample_ids,
                "features": torch.randn(len(rows), time, dim),
                "lengths": torch.full((len(rows),), time, dtype=torch.long),
                "feature_extractor": {"type": f"synthetic_{modality_name}"},
            },
            processed_root / f"{modality_name}.pt",
        )
    return raw_root, processed_root


def test_iemocap_loader_returns_three_sequence_modalities(tmp_path):
    raw_root, processed_root = _write_synthetic_iemocap_cache(tmp_path)
    dataset = load_dataset(_config(raw_root, processed_root), tmp_path)

    assert dataset["modality_names"] == ["audio", "video", "text"]
    assert dataset["modality_encoder_types"] == ["conv_gru", "gru", "gru"]
    assert dataset["modality_input_shapes"] == [[12, 4], [5, 6], [7, 8]]
    assert dataset["split_num_samples"] == {"train": 6, "validation": 2, "test": 2}
    assert dataset["label_mapping"] == {
        "angry": 0,
        "happy_or_excited": 1,
        "sad": 2,
        "neutral": 3,
    }
    for split_name in ("train", "validation", "test"):
        split = dataset[split_name]
        assert len(split["modalities"]) == 3
        assert len(split["modality_lengths"]) == 3
        for features, lengths in zip(split["modalities"], split["modality_lengths"]):
            assert int(features.shape[0]) == int(split["labels"].shape[0])
            assert int(lengths.shape[0]) == int(split["labels"].shape[0])
            time = torch.arange(features.shape[1]).unsqueeze(0)
            padded = time >= lengths.unsqueeze(1)
            assert torch.equal(features[padded], torch.zeros_like(features[padded]))


def test_iemocap_stage1_preserves_encoder_type_and_lengths(tmp_path):
    raw_root, processed_root = _write_synthetic_iemocap_cache(tmp_path)
    cfg = _config(raw_root, processed_root)
    cfg["partition"]["output_dir"] = str(tmp_path / "partition")

    info = run_stage1_partition(cfg, tmp_path)
    output_dir = Path(info["output_dir"])
    payloads = [
        torch.load(path, map_location="cpu")
        for path in sorted((output_dir / "train_clients").glob("client_*.pt"))
    ]
    assert [payload["encoder_type"] for payload in payloads] == [
        "conv_gru",
        "conv_gru",
        "gru",
        "gru",
        "gru",
        "gru",
    ]
    assert all(payload["sequence_lengths"] is not None for payload in payloads)
    evaluation = torch.load(output_dir / "test_multimodal.pt", map_location="cpu")
    assert set(evaluation["modality_lengths"]) == {"audio", "video", "text"}


def test_fedmultimodal_sequence_encoders_produce_common_activation_size():
    cfg = {
        "encoder_hidden_dim": 16,
        "model": {
            "encoder": {
                "conv_channels": [8, 12, 16],
                "kernel_size": 5,
                "dropout": 0.0,
            }
        },
    }
    audio_encoder = create_client_encoder(cfg, input_shape=[24, 4], encoder_type="conv_gru")
    video_encoder = create_client_encoder(cfg, input_shape=[5, 6], encoder_type="gru")
    audio = audio_encoder(torch.randn(3, 24, 4), torch.tensor([24, 20, 12]))
    video = video_encoder(torch.randn(3, 5, 6), torch.tensor([5, 4, 3]))
    assert audio.shape == (3, 16)
    assert video.shape == (3, 16)


def test_iemocap_loso_uses_unseen_test_session_and_dialog_disjoint_validation(tmp_path):
    raw_root, processed_root = _write_loso_iemocap_cache(tmp_path)
    cfg = _config(raw_root, processed_root)
    cfg["dataset"].update(
        {
            "split_strategy": "session_loso_5fold_group_validation_v1",
            "split_protocol": "iemocap_session_loso_fold2_v1",
            "train_sessions": [1, 3, 4, 5],
            "validation_sessions": [],
            "test_sessions": [2],
            "validation_folds": 4,
            "validation_fold_index": 0,
            "validation_seed": 101,
        }
    )

    dataset = load_dataset(cfg, tmp_path)
    assert dataset["split_num_samples"]["test"] == 32
    assert dataset["split_num_samples"]["train"] + dataset["split_num_samples"]["validation"] == 128
    assert dataset["split_metadata"]["test_sessions"] == [2]
    assert dataset["split_metadata"]["train_dialogs"] == 24
    assert dataset["split_metadata"]["validation_dialogs"] == 8


def test_temporal_and_attention_encoders_produce_configured_activation_size():
    temporal_cfg = {
        "encoder_hidden_dim": 16,
        "model": {
            "encoder": {
                "type": "temporal_conv_gru",
                "conv_channels": [8, 12],
                "kernel_sizes": [5, 3],
                "gru_hidden_dim": 12,
                "gru_layers": 2,
                "bidirectional": True,
                "pooling": "attention",
                "dropout": 0.0,
            }
        },
    }
    temporal = create_client_encoder(temporal_cfg, input_shape=[3, 32])
    assert temporal(torch.randn(4, 3, 32)).shape == (4, 16)

    sequence_cfg = {
        "encoder_hidden_dim": 16,
        "model": {
            "encoder": {"type": "gru"},
            "encoders": {
                "gru": {
                    "gru_layers": 2,
                    "bidirectional": True,
                    "pooling": "attention",
                    "dropout": 0.0,
                }
            },
        },
    }
    sequence = create_client_encoder(sequence_cfg, input_shape=[12, 6], encoder_type="gru")
    assert sequence(torch.randn(4, 12, 6), torch.tensor([12, 10, 8, 6])).shape == (4, 16)
