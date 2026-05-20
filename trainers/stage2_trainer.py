import torch

from clients.client_node import SplitClient
from clustering.kmeans_cluster import run_kmeans, evaluate_clustering
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

    def _validate_dataset_and_partition(self):
        # 1) Paired test-set sanity checks
        assert "modalities" in self.test_set and "labels" in self.test_set, "test_set must contain modalities and labels"
        assert len(self.test_set["modalities"]) == self.cfg["num_modalities"], (
            f"test modalities mismatch: expected {self.cfg['num_modalities']}, got {len(self.test_set['modalities'])}"
        )
        test_n = len(self.test_set["labels"])
        for m, x in enumerate(self.test_set["modalities"]):
            assert x.shape[0] == test_n, f"test modality {m} sample size mismatch: {x.shape[0]} vs {test_n}"

        # 2) Client partition sanity checks
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

    def cluster_clients(self):
        gt = [c.gt_cluster for c in self.clients]
        reps = [c.cluster_representation().cpu().numpy() for c in self.clients]
        pred = run_kmeans(reps, known_k=self.cfg["cluster"]["known_k"], seed=self.cfg["seed"])
        mapping, cm, acc, nmi, ari = evaluate_clustering(gt, pred, k=self.cfg["cluster"]["known_k"])

        print("ground-truth modality clusters:", gt)
        print("KMeans predicted clusters:", pred.tolist())
        print("cluster->modality majority mapping:", mapping)
        print("confusion matrix:\n", cm)
        print(f"cluster metrics: acc={acc:.4f}, nmi={nmi:.4f}, ari={ari:.4f}")

        # Protocol-consistent scheduling pools:
        # one pool per true modality cluster defined by controlled partitioning.
        # KMeans is used for diagnostics/evaluation only.
        modality_to_clients = {
            m: [c.client_id for c in self.clients_by_modality[m]]
            for m in range(self.cfg["num_modalities"])
        }
        print("scheduler modality pools (GT partition):", {m: len(v) for m, v in modality_to_clients.items()})
        return modality_to_clients

    def run(self):
        modality_to_clients = self.cluster_clients()
        scheduler = FairRandomFullModalityScheduler(modality_to_clients, seed=self.cfg["seed"])

        for round_idx in range(self.cfg["global_rounds"]):
            selected = scheduler.select()
            print(f"\n=== Global Round {round_idx} ===")
            print("selected clients:", selected)

            selected_clients = [self.clients_by_id[cid] for _, cid in sorted(selected.items())]
            selected_modalities = [c.modality_id for c in selected_clients]
            assert len(set(selected_modalities)) == self.cfg["num_modalities"], (
                f"selected clients are not full-modality: {selected_modalities}"
            )
            print("selected modality ids:", selected_modalities)
            for local_step in range(self.cfg["local_steps_per_round"]):
                print(f"-- Local Step {local_step}")

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
                    continue

                for client in selected_clients:
                    grad = out["grad_to_clients"][client.client_id]
                    client.backward_from_server(z_cache[client.client_id], grad)

        metrics = evaluate_paired_test(self.clients_by_modality, self.server, self.test_set, self.device)
        print("\n=== Paired multimodal test evaluation ===")
        print(f"test top1_acc={metrics['top1_acc']:.4f}, macro_f1={metrics['macro_f1']:.4f}")
        return metrics
