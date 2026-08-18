import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch

from MSL.data import Client
from MSL.discovery import build_fingerprints
from MSL.protocol import DATASET_PROTOCOLS
from experiments.common import build_experiment_config, apply_experiment_overrides, with_repeated_seed_split_signature


def _prepare_pca(fingerprints: np.ndarray, standardize: bool = True):
    fingerprints = np.asarray(fingerprints, dtype=np.float64)
    if fingerprints.ndim != 2 or fingerprints.shape[0] < 2 or fingerprints.shape[1] < 1:
        raise ValueError("fingerprints must be a 2D array with at least two clients and one feature.")
    transformed = StandardScaler().fit_transform(fingerprints) if standardize else fingerprints.copy()
    components = min(2, transformed.shape[0], transformed.shape[1])
    pca = PCA(n_components=components, svd_solver="full")
    coordinates = pca.fit_transform(transformed)
    if components == 1:
        coordinates = np.column_stack([coordinates[:, 0], np.zeros(coordinates.shape[0])])
        explained = [float(pca.explained_variance_ratio_[0]), 0.0]
    else:
        explained = [float(value) for value in pca.explained_variance_ratio_]
    return coordinates, explained


def _ellipse_parameters(points: np.ndarray, confidence: float):
    if len(points) < 3 or not 0.0 < confidence < 1.0:
        return None
    covariance = np.cov(points, rowvar=False)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 0.0)
    if values[-1] <= 1.0e-12:
        return None
    order = values.argsort()[::-1]
    values = values[order]
    vectors = vectors[:, order]
    radius = np.sqrt(-2.0 * np.log(1.0 - confidence))
    width, height = 2.0 * radius * np.sqrt(values)
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    return points.mean(axis=0), float(width), float(height), angle


def _plot_panel(ax, coordinates, labels, display_names, title, cfg, client_ids):
    from matplotlib.patches import Ellipse

    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
    markers = ["o", "s", "^", "D", "P", "X", "v", "<"]
    unique_labels = sorted({int(value) for value in labels})
    for index, label in enumerate(unique_labels):
        mask = np.asarray(labels, dtype=int) == label
        color = palette[index % len(palette)]
        marker = markers[index % len(markers)]
        name = display_names.get(int(label), str(label))
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=42,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            alpha=0.92,
            label=name,
            zorder=3,
        )
        if bool(cfg.get("show_ellipses", True)):
            params = _ellipse_parameters(coordinates[mask], float(cfg.get("ellipse_confidence", 0.95)))
            if params is not None:
                center, width, height, angle = params
                ax.add_patch(
                    Ellipse(
                        center,
                        width,
                        height,
                        angle=angle,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1.0,
                        alpha=0.12,
                        zorder=1,
                    )
                )
        if bool(cfg.get("show_client_ids", False)):
            for x, y, client_id in zip(coordinates[mask, 0], coordinates[mask, 1], np.asarray(client_ids)[mask]):
                ax.annotate(str(client_id), (x, y), xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.35)
    ax.legend(frameon=True, fontsize=7.5, title_fontsize=8, loc="best")


def write_fingerprint_pca_figure(
    fingerprints,
    client_ids,
    true_clusters,
    pred_clusters,
    output_dir: Path,
    dataset_name: str,
    true_cluster_names: dict | None = None,
    visualization_cfg: dict | None = None,
):
    cfg = dict(visualization_cfg or {})
    method = str(cfg.get("method", "pca")).strip().lower()
    if method != "pca":
        raise ValueError(f"fingerprint_visualization.method must be 'pca', got {method!r}.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = np.asarray(fingerprints, dtype=np.float32)
    client_ids = np.asarray(client_ids, dtype=str)
    true_clusters = np.asarray(true_clusters, dtype=int)
    pred_clusters = np.asarray(pred_clusters, dtype=int)
    n = fingerprints.shape[0]
    if not (len(client_ids) == len(true_clusters) == len(pred_clusters) == n):
        raise ValueError("fingerprints, client_ids, true_clusters, and pred_clusters must have equal row counts.")

    coordinates, explained = _prepare_pca(fingerprints, bool(cfg.get("standardize", True)))
    true_names = {int(key): str(value) for key, value in (true_cluster_names or {}).items()}
    for label in sorted(set(true_clusters.tolist())):
        true_names.setdefault(int(label), f"Modality {label}")
    pred_names = {int(label): f"Cluster {label}" for label in sorted(set(pred_clusters.tolist()))}

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "MSL_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    _plot_panel(axes[0], coordinates, true_clusters, true_names, "(a) Ground-truth modality audit", cfg, client_ids)
    _plot_panel(axes[1], coordinates, pred_clusters, pred_names, "(b) Predicted cluster assignment", cfg, client_ids)
    fig.suptitle(f"Pre-clustering PCA of Client Fingerprints — {dataset_name}", fontsize=11)

    pdf_path = output_dir / "fingerprint_pca.pdf"
    png_path = output_dir / "fingerprint_pca.png"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(png_path, format="png", dpi=int(cfg.get("png_dpi", 600)), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    np.savez_compressed(
        output_dir / "fingerprints.npz",
        fingerprints=fingerprints,
        client_ids=client_ids,
        true_cluster=true_clusters,
        pred_cluster=pred_clusters,
        pca_coordinates=coordinates.astype(np.float32),
    )
    metadata = {
        "visualization": "pre_clustering_client_fingerprint_pca",
        "coordinate_input": "fingerprints_only",
        "labels_used_for_pca": False,
        "true_cluster_usage": "post_hoc_audit_coloring_only",
        "pred_cluster_usage": "post_hoc_audit_coloring_only",
        "dataset": str(dataset_name),
        "num_clients": int(n),
        "raw_fingerprint_dim": int(fingerprints.shape[1]),
        "standardized_before_pca": bool(cfg.get("standardize", True)),
        "explained_variance_ratio": explained,
        "show_ellipses": bool(cfg.get("show_ellipses", True)),
        "ellipse_confidence": float(cfg.get("ellipse_confidence", 0.95)),
        "png_dpi": int(cfg.get("png_dpi", 600)),
        "pdf_format": "vector",
    }
    metadata_path = output_dir / "fingerprint_pca_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return {"pdf": pdf_path, "png": png_path, "npz": output_dir / "fingerprints.npz", "metadata": metadata_path}


def _read_assignments(path: Path, column: str):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return {row["client_id"]: int(row[column]) for row in csv.DictReader(f)}


def rebuild_fingerprint_figure(cfg: dict, clients_dir: Path, discovery_dir: Path, device: torch.device):
    from MSL.models import create_client_encoder

    clients_dir = Path(clients_dir)
    discovery_dir = Path(discovery_dir)
    payloads = [torch.load(path, map_location="cpu") for path in sorted((clients_dir / "train_clients").glob("client_*.pt"))]
    if not payloads:
        raise FileNotFoundError(f"No client payloads found under {clients_dir / 'train_clients'}.")
    clients = [Client.from_payload(payload) for payload in payloads]
    encoders = {}
    for client in clients:
        saved = torch.load(discovery_dir / "pretrained_encoders" / f"{client.client_id}_encoder.pt", map_location="cpu")
        encoder = create_client_encoder(cfg, input_shape=client.input_shape, encoder_type=client.encoder_type).to(device)
        encoder.load_state_dict(saved["state_dict"])
        encoders[client.client_id] = encoder
    fingerprints = build_fingerprints(clients, encoders, cfg, device)
    true_map = _read_assignments(discovery_dir / "true_cluster.csv", "true_cluster")
    pred_map = _read_assignments(discovery_dir / "pred_cluster.csv", "pred_cluster")
    names = {}
    for payload in payloads:
        names[int(payload["hidden_modality_id"])] = str(payload.get("hidden_modality_name", f"Modality {payload['hidden_modality_id']}"))
    client_ids = [client.client_id for client in clients]
    return write_fingerprint_pca_figure(
        fingerprints,
        client_ids,
        [true_map[client_id] for client_id in client_ids],
        [pred_map[client_id] for client_id in client_ids],
        discovery_dir,
        cfg.get("dataset", {}).get("name", cfg.get("dataset", {}).get("type", "dataset")),
        true_cluster_names=names,
        visualization_cfg=cfg.get("fingerprint_visualization", {}),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create publication-quality PCA plots of modality discovery client fingerprints.")
    parser.add_argument("--dataset", choices=tuple(DATASET_PROTOCOLS), required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fingerprint-type", choices=["encoder", "signal", "hybrid"])
    parser.add_argument("--clients-dir", required=True)
    parser.add_argument("--discovery-dir", required=True)
    args = parser.parse_args(argv)
    cfg = build_experiment_config(dataset_type=args.dataset, seed=args.seed, device=args.device)
    cfg = apply_experiment_overrides(cfg, fold=args.fold)
    if args.fold is None and DATASET_PROTOCOLS[str(args.dataset)]["fold_count"] is None:
        cfg = with_repeated_seed_split_signature(cfg, args.seed)
    if args.fingerprint_type is not None:
        cfg["fingerprint"] = {**dict(cfg.get("fingerprint", {})), "type": str(args.fingerprint_type)}
    outputs = rebuild_fingerprint_figure(cfg, Path(args.clients_dir), Path(args.discovery_dir), torch.device(args.device))
    for key, path in outputs.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
