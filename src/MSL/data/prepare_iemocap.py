import argparse
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]

import torch

from .iemocap import IEMOCAP_LABEL_MAPPING, load_manifest


ANNOTATION_PATTERN = re.compile(
    r"^\[\s*([0-9.]+)\s*-\s*([0-9.]+)\]\s+(\S+)\s+(\S+)\s+\["
)
TRANSCRIPT_PATTERN = re.compile(r"^(\S+)\s+\[[^]]+\]:\s*(.*)$")


def _resolve(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_transcripts(root: Path, required_ids: set[str]) -> dict[str, str]:
    transcripts = {}
    for session_id in range(1, 6):
        directory = root / f"Session{session_id}" / "dialog" / "transcriptions"
        for path in sorted(directory.glob("*.txt")):
            if path.name.startswith("._"):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = TRANSCRIPT_PATTERN.match(line.strip())
                if match:
                    utterance_id, transcript = match.groups()
                    if utterance_id not in required_ids:
                        continue
                    if utterance_id in transcripts:
                        raise ValueError(f"Duplicate IEMOCAP transcript ID: {utterance_id}")
                    transcripts[utterance_id] = transcript.strip()
    return transcripts


def build_manifest(root: Path, output_path: Path) -> dict:
    required_ids: set[str] = set()
    for session_id in range(1, 6):
        directory = root / f"Session{session_id}" / "dialog" / "EmoEvaluation"
        for path in sorted(directory.glob("*.txt")):
            if path.name.startswith("._"):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = ANNOTATION_PATTERN.match(line.strip())
                if match and match.group(4) in IEMOCAP_LABEL_MAPPING:
                    required_ids.add(match.group(3))
    transcripts = _read_transcripts(root, required_ids)
    rows = []
    for session_id in range(1, 6):
        directory = root / f"Session{session_id}" / "dialog" / "EmoEvaluation"
        for path in sorted(directory.glob("*.txt")):
            if path.name.startswith("._"):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                match = ANNOTATION_PATTERN.match(line.strip())
                if not match:
                    continue
                start, end, utterance_id, emotion = match.groups()
                if emotion not in IEMOCAP_LABEL_MAPPING:
                    continue
                dialog_id = utterance_id.rsplit("_", 1)[0]
                audio_path = (
                    root
                    / f"Session{session_id}"
                    / "sentences"
                    / "wav"
                    / dialog_id
                    / f"{utterance_id}.wav"
                )
                video_path = (
                    root
                    / f"Session{session_id}"
                    / "dialog"
                    / "avi"
                    / "DivX"
                    / f"{dialog_id}.avi"
                )
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing IEMOCAP utterance audio: {audio_path}")
                if not video_path.exists():
                    raise FileNotFoundError(f"Missing IEMOCAP dialog video: {video_path}")
                if utterance_id not in transcripts:
                    raise ValueError(f"Missing IEMOCAP transcript: {utterance_id}")
                speaker = utterance_id.rsplit("_", 1)[1][0].upper()
                rows.append(
                    {
                        "utterance_id": utterance_id,
                        "session_id": session_id,
                        "dialog_id": dialog_id,
                        "speaker": speaker,
                        "start_time": float(start),
                        "end_time": float(end),
                        "original_emotion": emotion,
                        "label": int(IEMOCAP_LABEL_MAPPING[emotion]),
                        "transcript": transcripts[utterance_id],
                        "audio_path": str(audio_path.relative_to(root)),
                        "video_path": str(video_path.relative_to(root)),
                        "scenario_type": "improvised" if "_impro" in dialog_id else "scripted",
                    }
                )
    if len(rows) != 5531:
        raise ValueError(f"Expected 5531 four-class IEMOCAP utterances, found {len(rows)}.")
    if len({row["utterance_id"] for row in rows}) != len(rows):
        raise ValueError("IEMOCAP manifest contains duplicate utterance IDs.")
    payload = {
        "dataset": "iemocap",
        "variant": "full",
        "task": "emotion_4class",
        "label_protocol": "ang_hap_exc_sad_neu_v1",
        "num_samples": len(rows),
        "samples": rows,
    }
    _write_json(output_path, payload)
    return payload


def extract_audio_features(root: Path, manifest: dict, output_path: Path, max_frames: int) -> None:
    import torchaudio

    transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={
            "n_fft": 400,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 64,
            "center": False,
        },
    )
    features = []
    lengths = []
    for index, row in enumerate(manifest["samples"], start=1):
        waveform, sample_rate = torchaudio.load(root / row["audio_path"])
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sequence = transform(waveform).squeeze(0).transpose(0, 1)
        sequence = torch.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        if int(sequence.shape[0]) > int(max_frames):
            sequence = torch.nn.functional.adaptive_avg_pool1d(
                sequence.transpose(0, 1).unsqueeze(0), int(max_frames)
            ).squeeze(0).transpose(0, 1)
        length = int(sequence.shape[0])
        padded = torch.zeros(int(max_frames), int(sequence.shape[1]), dtype=torch.float32)
        padded[:length] = sequence
        features.append(padded.to(dtype=torch.float16))
        lengths.append(length)
        if index % 500 == 0:
            print(f"audio_features={index}/{len(manifest['samples'])}", flush=True)
    torch.save(
        {
            "sample_ids": [row["utterance_id"] for row in manifest["samples"]],
            "features": torch.stack(features),
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "feature_extractor": {
                "type": "mfcc",
                "sample_rate": 16000,
                "n_mfcc": 40,
                "window_ms": 25,
                "hop_ms": 10,
                "max_frames": int(max_frames),
                "long_sequence_policy": "adaptive_average_pool",
            },
        },
        output_path,
    )


def extract_text_features(
    manifest: dict,
    output_path: Path,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
    max_tokens: int,
) -> None:
    from transformers import AutoModel, AutoTokenizer

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
    rows = manifest["samples"]
    feature_batches = []
    length_batches = []
    with torch.no_grad():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            tokens = tokenizer(
                [row["transcript"] for row in batch_rows],
                padding="max_length",
                truncation=True,
                max_length=int(max_tokens),
                return_tensors="pt",
            )
            attention_mask = tokens["attention_mask"]
            model_inputs = {key: value.to(device) for key, value in tokens.items()}
            hidden = model(**model_inputs).last_hidden_state.detach().cpu()
            hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            feature_batches.append(hidden.to(dtype=torch.float16))
            length_batches.append(attention_mask.sum(dim=1).to(dtype=torch.long))
            print(f"text_features={min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    torch.save(
        {
            "sample_ids": [row["utterance_id"] for row in rows],
            "features": torch.cat(feature_batches, dim=0),
            "lengths": torch.cat(length_batches, dim=0),
            "feature_extractor": {
                "type": "distilbert_token_embedding",
                "model": model_name,
                "max_tokens": int(max_tokens),
                "frozen": True,
            },
        },
        output_path,
    )


def _read_utterance_frames(capture, row: dict, num_frames: int):
    import cv2
    from PIL import Image

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise ValueError(f"Invalid video FPS for {row['video_path']}: {fps}")
    start_frame = max(0, int(round(float(row["start_time"]) * fps)))
    end_frame = max(start_frame, int(round(float(row["end_time"]) * fps)) - 1)
    if num_frames == 1:
        indices = [(start_frame + end_frame) // 2]
    else:
        indices = torch.linspace(start_frame, end_frame, steps=int(num_frames)).round().to(dtype=torch.long).tolist()
    dialog_actor = row["dialog_id"].split("_", 1)[0][-1].upper()
    use_left_half = dialog_actor == row["speaker"].upper()
    unique_indices = set(int(value) for value in indices)
    current_frame = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    first_frame = min(unique_indices)
    long_forward_gap = first_frame - current_frame > int(round(3.0 * fps))
    if current_frame < 0 or current_frame > first_frame or long_forward_gap:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first_frame)
        current_frame = first_frame
    frames_by_index = {}
    last_decoded_frame = None
    while current_frame <= max(unique_indices):
        is_sampled_frame = current_frame in unique_indices
        if is_sampled_frame:
            ok, frame = capture.read()
        else:
            ok = capture.grab()
            frame = None
        if not ok:
            if last_decoded_frame is None:
                raise RuntimeError(f"Failed to decode {row['video_path']} frame {current_frame}.")
            break
        if is_sampled_frame:
            last_decoded_frame = frame
            height, width = frame.shape[:2]
            content = frame[int(height * 0.20) : int(height * 0.80)]
            speaker_frame = content[:, : width // 2] if use_left_half else content[:, width // 2 :]
            rgb = cv2.cvtColor(speaker_frame, cv2.COLOR_BGR2RGB)
            frames_by_index[current_frame] = Image.fromarray(rgb)
        current_frame += 1
    missing_indices = sorted(unique_indices - set(frames_by_index))
    if missing_indices:
        height, width = last_decoded_frame.shape[:2]
        content = last_decoded_frame[int(height * 0.20) : int(height * 0.80)]
        speaker_frame = content[:, : width // 2] if use_left_half else content[:, width // 2 :]
        fallback_image = Image.fromarray(cv2.cvtColor(speaker_frame, cv2.COLOR_BGR2RGB))
        for frame_idx in missing_indices:
            frames_by_index[frame_idx] = fallback_image.copy()
    return [frames_by_index[int(frame_idx)] for frame_idx in indices], len(missing_indices)


def extract_video_features(
    root: Path,
    manifest: dict,
    output_path: Path,
    progress_path: Path,
    device: torch.device,
    num_frames: int,
) -> None:
    import cv2
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model_name = "mobilevit_xs.cvnets_in1k"
    model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
    model = model.to(device).eval()
    transform = create_transform(**resolve_model_data_config(model), is_training=False)
    rows = manifest["samples"]
    completed_ids = []
    feature_rows = []
    tail_frame_fallback_count = 0
    if progress_path.exists():
        progress = torch.load(progress_path, map_location="cpu")
        completed_ids = [str(value) for value in progress.get("sample_ids", [])]
        feature_rows = list(progress.get("features", []))
        tail_frame_fallback_count = int(progress.get("tail_frame_fallback_count", 0))
        expected_prefix = [row["utterance_id"] for row in rows[: len(completed_ids)]]
        if completed_ids != expected_prefix or len(feature_rows) != len(completed_ids):
            raise ValueError(f"Invalid IEMOCAP video progress cache: {progress_path}")
        print(f"resume_video_features={len(completed_ids)}/{len(rows)}", flush=True)

    capture = None
    current_video_path = None
    try:
        for index in range(len(completed_ids), len(rows)):
            row = rows[index]
            video_path = root / row["video_path"]
            if video_path != current_video_path:
                if capture is not None:
                    capture.release()
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f"Failed to open IEMOCAP video: {video_path}")
                current_video_path = video_path
            frames, fallback_count = _read_utterance_frames(capture, row, int(num_frames))
            tail_frame_fallback_count += int(fallback_count)
            batch = torch.stack([transform(frame) for frame in frames]).to(device)
            with torch.no_grad():
                embedding = model(batch).detach().cpu().to(dtype=torch.float16)
            feature_rows.append(embedding)
            completed_ids.append(row["utterance_id"])
            if (index + 1) % 25 == 0 or index + 1 == len(rows):
                torch.save(
                    {
                        "sample_ids": completed_ids,
                        "features": feature_rows,
                        "tail_frame_fallback_count": tail_frame_fallback_count,
                    },
                    progress_path,
                )
                print(f"video_features={index + 1}/{len(rows)}", flush=True)
    finally:
        if capture is not None:
            capture.release()

    torch.save(
        {
            "sample_ids": completed_ids,
            "features": torch.stack(feature_rows),
            "lengths": torch.full((len(feature_rows),), int(num_frames), dtype=torch.long),
            "feature_extractor": {
                "type": "mobilevit_frame_embedding",
                "model": model_name,
                "frames_per_utterance": int(num_frames),
                "speaker_crop": "dialog_actor_left_else_right",
                "vertical_content_crop": [0.20, 0.80],
                "tail_frame_fallback_count": int(tail_frame_fallback_count),
                "tail_frame_fallback_policy": "repeat_last_decoded_frame",
                "frozen": True,
            },
        },
        output_path,
    )


def write_metadata(processed_root: Path) -> None:
    required = [processed_root / "manifest.json"] + [
        processed_root / f"{name}.pt" for name in ("audio", "video", "text")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"metadata_not_written_missing={missing}")
        return
    manifest = load_manifest(processed_root)
    modalities = {}
    for name in ("audio", "video", "text"):
        payload = torch.load(processed_root / f"{name}.pt", map_location="cpu")
        modalities[name] = {
            "shape": [int(value) for value in payload["features"].shape],
            "length_min": int(payload["lengths"].min().item()),
            "length_max": int(payload["lengths"].max().item()),
            "feature_extractor": payload["feature_extractor"],
        }
    _write_json(
        processed_root / "metadata.json",
        {
            "dataset": "iemocap",
            "variant": "full",
            "feature_recipe": "mfcc_mobilevit_xs_distilbert_v1",
            "num_samples": int(manifest["num_samples"]),
            "modalities": modalities,
        },
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Prepare IEMOCAP Full audio/video/text feature caches.")
    parser.add_argument(
        "--root",
        default="./local/datasets/IEMOCAP/IEMOCAP_full/IEMOCAP_full_release",
    )
    parser.add_argument(
        "--output-dir",
        default="./local/datasets/IEMOCAP/processed/mfcc_mobilevit_xs_distilbert_v1",
    )
    parser.add_argument(
        "--cache-dir",
        default="./local/datasets/IEMOCAP/model_cache",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["manifest", "audio", "text", "video"],
        default=["manifest", "audio", "text", "video"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--max-audio-frames", type=int, default=1200)
    parser.add_argument("--max-text-tokens", type=int, default=64)
    parser.add_argument("--video-frames", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = _resolve(args.root)
    processed_root = _resolve(args.output_dir)
    cache_dir = _resolve(args.cache_dir)
    processed_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    if not root.exists():
        raise FileNotFoundError(f"IEMOCAP full root not found: {root}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this process.")

    manifest_path = processed_root / "manifest.json"
    if "manifest" in args.steps:
        if manifest_path.exists() and not args.force:
            print(f"skip_existing={manifest_path}")
        else:
            build_manifest(root, manifest_path)
            print(f"saved={manifest_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest required by feature extraction: {manifest_path}")
    manifest = load_manifest(processed_root)

    actions = {
        "audio": lambda path: extract_audio_features(root, manifest, path, args.max_audio_frames),
        "text": lambda path: extract_text_features(
            manifest,
            path,
            cache_dir,
            device,
            args.text_batch_size,
            args.max_text_tokens,
        ),
        "video": lambda path: extract_video_features(
            root,
            manifest,
            path,
            processed_root / "video_progress.pt",
            device,
            args.video_frames,
        ),
    }
    for step in ("audio", "text", "video"):
        if step not in args.steps:
            continue
        output_path = processed_root / f"{step}.pt"
        if output_path.exists() and not args.force:
            print(f"skip_existing={output_path}")
            continue
        actions[step](output_path)
        print(f"saved={output_path}")
    write_metadata(processed_root)


if __name__ == "__main__":
    main()
