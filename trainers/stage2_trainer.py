import json
from datetime import datetime
from pathlib import Path
import torch

from clients.client_node import SplitClient
from clustering.cluster import run_kmeans, evaluate_clustering
from trainers.schedulers import FairRandomFullModalityScheduler
from server.server_core import SplitServer
from server.evaluation import evaluate_paired_test


class Stage2Trainer:
    def __init__(self, cfg, train_clients_raw, test_set, device):
        self.cfg = cfg
        self.device = device
        self.test_set = test_set
        self.clients = [SplitClient(c, cfg, device) for c in train_clients_raw]

        self.clients_by_id = {c.client_id: c for c in self.clients}
        self.clients_by_modality = {m: [] for m in range(cfg["num_modalities"])}
        for c in self.clients:
            self.clients_by_modality[c.modality_id].append(c)

        self.server = SplitServer(cfg, device).to(device)
        self._validate_dataset_and_partition()
        self._init_logging_state()

    def _init_logging_state(self):
        exp_name = self.cfg.get("experiment_name", "").strip()
        if not exp_name:
            exp_name = f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = exp_name
        self.results_dir = Path("experiments/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = []
        self.best_test_acc = -1.0
        self.best_macro_f1 = -1.0

    def _validate_dataset_and_partition(self):
        assert "modalities" in self.test_set and "labels" in self.test_set, "test_set must contain modalities and labels"
        assert len(self.test_set["modalities"]) == self.cfg["num_modalities"], (
            f"test modalities mismatch: expected {self.cfg['num_modalities']}, got {len(self.test_set['modalities'])}"
        )
        test_n = len(self.test_set["labels"])
        for m, x in enumerate(self.test_set["modalities"]):
            assert x.shape[0] == test_n, f"test modality {m} sample size mismatch: {x.shape[0]} vs {test_n}"

        expected_clients = self.cfg["num_modalities"] * self.cfg["clients_per_modality"]
        assert len(self.clients) == expected_clients, f"client count mismatch: {len(self.clients)} vs {expected_clients}"

        per_modality_count = {m: len(self.clients_by_modality[m]) for m in self.clients_by_modality}
        for m in range(self.cfg["num_modalities"]):
            assert per_modality_count[m] == self.cfg["clients_per_modality"], (
                f"modality {m} client count mismatch: {per_modality_count[m]} vs {self.cfg['clients_per_modality']}"
            )

        min_labels_required = max(2, int(self.cfg["num_classes"] * self.cfg["min_labels_per_client_ratio"]))
        sample_sizes = []
        for c in self.clients:
            unique_labels = torch.unique(c.y).numel()
            assert unique_labels >= min_labels_required, (
                f"{c.client_id} unique labels {unique_labels} < required {min_labels_required}"
            )
            sample_sizes.append(int(c.y.shape[0]))
        assert len(set(sample_sizes)) == 1, f"client sample sizes are not equal: {sorted(set(sample_sizes))}"

        print("validation(dataset/partition): passed")
        print(f"test paired samples: {test_n}")
        print(f"clients per modality: {per_modality_count}")
        print(f"min unique labels per client required: {min_labels_required}")

    def _run_stage1_autoencoder_pretraining(self):
        stage1_cfg = self.cfg.get("stage1", {}).get("autoencoder_pretrain", {})
        enabled = bool(stage1_cfg.get("enabled", True))
        epochs = int(stage1_cfg.get("epochs", 5))
        batch_size = int(stage1_cfg.get("batch_size", 64))
        lr = float(stage1_cfg.get("lr", 0.001))
        weight_decay = float(stage1_cfg.get("weight_decay", 0.0))
        max_samples = stage1_cfg.get("max_samples", None)

        print(
            "stage1 autoencoder pretraining: "
            f"enabled={enabled}, epochs={epochs}, batch_size={batch_size}, lr={lr}, "
            f"weight_decay={weight_decay}, max_samples={max_samples}"
        )

        losses = []
        for client in self.clients:
            out = client.pretrain_autoencoder(
                enabled=enabled,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                max_samples=max_samples,
            )
            if out["avg_recon_loss"] is not None:
                losses.append(out["avg_recon_loss"])
        if losses:
            print(f"stage1 pretrain avg recon loss across clients: {sum(losses) / len(losses):.6f}")

    def cluster_clients(self):
        self._run_stage1_autoencoder_pretraining()

        gt = [c.gt_cluster for c in self.clients]

        clustering_cfg = self.cfg.get("clustering", {})
        fp_max_samples = clustering_cfg.get("fingerprint_max_samples", None)
        if fp_max_samples is None:
            fp_max_samples = self.cfg.get("stage1", {}).get("autoencoder_pretrain", {}).get("max_samples", None)
        fp_normalize = bool(clustering_cfg.get("fingerprint_normalize", False))

        reps = [
            c.cluster_representation(max_samples=fp_max_samples, normalize=fp_normalize).cpu().numpy()
            for c in self.clients
        ]
        if len(reps) > 0:
            print(f"fingerprint shape (single client): {reps[0].shape}")
            print(f"kmeans input matrix shape: ({len(reps)}, {len(reps[0])})")

        pred = run_kmeans(reps, known_k=self.cfg["cluster"]["known_k"], seed=self.cfg["seed"])
        mapping, cm, acc, nmi, ari = evaluate_clustering(gt, pred, k=self.cfg["cluster"]["known_k"])

        print("ground-truth modality clusters:", gt)
        print("KMeans predicted clusters:", pred.tolist())
        print("cluster->modality majority mapping:", mapping)
        print("confusion matrix:\n", cm)
        print(f"cluster metrics: acc={acc:.4f}, nmi={nmi:.4f}, ari={ari:.4f}")

        gt_pools = {m: [c.client_id for c in self.clients_by_modality[m]] for m in range(self.cfg["num_modalities"])}
        pred_cluster_pools = {m: [] for m in range(self.cfg["num_modalities"])}
        for c, p in zip(self.clients, pred):
            pred_cluster_pools[int(p)].append(c.client_id)
        pred_mapped_modality_pools = {m: [] for m in range(self.cfg["num_modalities"])}
        for cluster_id, client_ids in pred_cluster_pools.items():
            mapped_modality = mapping.get(cluster_id, -1)
            if mapped_modality in pred_mapped_modality_pools:
                pred_mapped_modality_pools[mapped_modality].extend(client_ids)
        print("scheduler GT modality pools:", {m: len(v) for m, v in gt_pools.items()})
        print("scheduler KMeans cluster pools:", {m: len(v) for m, v in pred_cluster_pools.items()})
        print("scheduler KMeans mapped modality pools:", {m: len(v) for m, v in pred_mapped_modality_pools.items()})
        return {
            "gt_pools": gt_pools,
            "pred_cluster_pools": pred_cluster_pools,
            "pred_mapped_modality_pools": pred_mapped_modality_pools,
            "cluster_acc": float(acc),
            "nmi": float(nmi),
            "ari": float(ari),
        }

    def _append_round_log(self, round_idx, round_stats, metrics, cluster_meta):
        attempted = round_stats["attempted_local_steps"]
        effective = round_stats["effective_local_steps"]
        skipped = round_stats["skipped_local_steps"]
        success_rate = effective / attempted if attempted > 0 else 0.0
        avg_batch = sum(round_stats["semantic_batch_sizes"]) / max(1, len(round_stats["semantic_batch_sizes"]))
        avg_common = sum(round_stats["common_label_counts"]) / max(1, len(round_stats["common_label_counts"]))
        avg_total = sum(round_stats["loss_total"]) / max(1, len(round_stats["loss_total"]))
        avg_cls = sum(round_stats["loss_cls"]) / max(1, len(round_stats["loss_cls"]))
        avg_align = sum(round_stats["loss_align"]) / max(1, len(round_stats["loss_align"]))
        avg_proto = sum(round_stats["loss_proto"]) / max(1, len(round_stats["loss_proto"]))
        grad_rate = sum(round_stats["grad_non_null_ratio"]) / max(1, len(round_stats["grad_non_null_ratio"]))

        row = {
            "global_round": int(round_idx),
            "effective_local_steps": int(effective),
            "attempted_local_steps": int(attempted),
            "skipped_local_steps": int(skipped),
            "common_label_success_rate": float(success_rate),
            "average_semantic_batch_size": float(avg_batch),
            "average_common_label_count": float(avg_common),
            "avg_total_loss": float(avg_total),
            "avg_cls_loss": float(avg_cls),
            "avg_align_loss": float(avg_align),
            "avg_proto_loss": float(avg_proto),
            "test_acc": float(metrics["acc"]),
            "test_macro_f1": float(metrics["macro_f1"]),
            "gradient_non_null_rate": float(grad_rate),
            "use_oracle_clusters_for_training": bool(self.cfg["clustering"]["use_oracle_clusters_for_training"]),
            "lambda_align": float(self.cfg.get("lambda_align", self.cfg.get("lambda_supcon", 0.0))),
            "scheduler_type": self.cfg.get("scheduler", {}).get("name", "fair_random_full_modality"),
            "clustering_method": self.cfg.get("cluster", {}).get("method", "kmeans"),
            "cluster_acc": float(cluster_meta["cluster_acc"]),
            "cluster_nmi": float(cluster_meta["nmi"]),
            "cluster_ari": float(cluster_meta["ari"]),
        }
        self.train_log.append(row)
        print(
            f"round stats: attempted={attempted}, effective={effective}, skipped={skipped}, "
            f"success={success_rate:.4f}, avg_sem_batch={avg_batch:.2f}, avg_common_labels={avg_common:.2f}, "
            f"grad_non_null={grad_rate:.4f}"
        )
        if attempted > 0 and (skipped / attempted) > 0.3:
            print("warning: skipped_local_steps ratio is high (>30%).")

    def _save_logs(self):
        train_log_path = self.results_dir / f"{self.experiment_name}_train_log.json"
        summary_path = self.results_dir / f"{self.experiment_name}_summary.json"
        train_log_path.write_text(json.dumps(self.train_log, indent=2), encoding="utf-8")

        total_attempted = sum(x["attempted_local_steps"] for x in self.train_log)
        total_effective = sum(x["effective_local_steps"] for x in self.train_log)
        total_skipped = sum(x["skipped_local_steps"] for x in self.train_log)
        final_acc = self.train_log[-1]["test_acc"] if self.train_log else 0.0
        final_f1 = self.train_log[-1]["test_macro_f1"] if self.train_log else 0.0
        summary = {
            "best_test_acc": float(self.best_test_acc if self.best_test_acc >= 0 else final_acc),
            "best_macro_f1": float(self.best_macro_f1 if self.best_macro_f1 >= 0 else final_f1),
            "final_test_acc": float(final_acc),
            "final_macro_f1": float(final_f1),
            "total_effective_local_steps": int(total_effective),
            "total_skipped_local_steps": int(total_skipped),
            "overall_common_label_success_rate": float(total_effective / total_attempted if total_attempted > 0 else 0.0),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved train log: {train_log_path}")
        print(f"saved summary: {summary_path}")

    def run(self):
        cluster_meta = self.cluster_clients()
        use_oracle = bool(self.cfg["clustering"]["use_oracle_clusters_for_training"])
        pools = cluster_meta["gt_pools"] if use_oracle else cluster_meta["pred_mapped_modality_pools"]
        if not use_oracle:
            missing = [m for m, ids in pools.items() if len(ids) == 0]
            if missing:
                raise RuntimeError(
                    "KMeans cluster-map training cannot form full-modality scheduling pools. "
                    f"Missing modality pools: {missing}. "
                    f"cluster_acc={cluster_meta['cluster_acc']:.4f}, nmi={cluster_meta['nmi']:.4f}, ari={cluster_meta['ari']:.4f}"
                )
        scheduler = FairRandomFullModalityScheduler(pools, seed=self.cfg["seed"])

        for round_idx in range(self.cfg["global_rounds"]):
            selected = scheduler.select()
            print(f"\n=== Global Round {round_idx} ===")
            print("selected clients:", selected)

            selected_clients = [self.clients_by_id[cid] for _, cid in sorted(selected.items())]
            selected_modalities = [c.modality_id for c in selected_clients]
            if len(set(selected_modalities)) != self.cfg["num_modalities"]:
                raise RuntimeError(
                    "Selected clients do not form a full-modality set. "
                    f"selected modalities={selected_modalities}, selected={selected}, "
                    f"use_oracle={use_oracle}"
                )
            print("selected modality ids:", selected_modalities)

            round_stats = {
                "attempted_local_steps": 0,
                "effective_local_steps": 0,
                "skipped_local_steps": 0,
                "semantic_batch_sizes": [],
                "common_label_counts": [],
                "loss_total": [],
                "loss_cls": [],
                "loss_align": [],
                "loss_proto": [],
                "grad_non_null_ratio": [],
            }

            for local_step in range(self.cfg["local_steps_per_round"]):
                print(f"-- Local Step {local_step}")
                round_stats["attempted_local_steps"] += 1

                payloads = []
                z_cache = {}
                for client in selected_clients:
                    x, y, _ = client.sample_batch()
                    batch_label_dist = torch.bincount(y, minlength=self.cfg["num_classes"]).tolist()
                    print(f"batch labels [{client.client_id}]: {batch_label_dist}")
                    z_client, z_server = client.forward_to_server(x)
                    payloads.append(
                        {
                            "client_id": client.client_id,
                            "modality_id": client.modality_id,
                            "y": y,
                            "z_server": z_server,
                        }
                    )
                    z_cache[client.client_id] = z_client

                out = self.server.train_step(payloads)
                if out is None:
                    print("common labels: []")
                    print("semantic batch size: 0")
                    round_stats["skipped_local_steps"] += 1
                    continue

                round_stats["effective_local_steps"] += 1
                round_stats["semantic_batch_sizes"].append(out["semantic_batch_size"])
                round_stats["common_label_counts"].append(out["common_label_count"])
                round_stats["loss_total"].append(out["loss_total"])
                round_stats["loss_cls"].append(out["loss_cls"])
                round_stats["loss_align"].append(out["loss_align"])
                round_stats["loss_proto"].append(out["loss_proto"])
                round_stats["grad_non_null_ratio"].append(out["grad_non_null_ratio"])

                for client in selected_clients:
                    grad = out["grad_to_clients"][client.client_id]
                    client.backward_update(z_cache[client.client_id], grad)

            metrics = evaluate_paired_test(self.clients_by_modality, self.server, self.test_set, self.device)
            self.best_test_acc = max(self.best_test_acc, metrics["acc"])
            self.best_macro_f1 = max(self.best_macro_f1, metrics["macro_f1"])
            self._append_round_log(round_idx, round_stats, metrics, cluster_meta)

        self._save_logs()
        final_metrics = evaluate_paired_test(self.clients_by_modality, self.server, self.test_set, self.device)
        print("\n=== Paired multimodal test evaluation ===")
        print(f"test top1_acc={final_metrics['top1_acc']:.4f}, macro_f1={final_metrics['macro_f1']:.4f}")
        return final_metrics
