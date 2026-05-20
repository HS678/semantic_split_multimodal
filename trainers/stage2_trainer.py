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

        cluster_to_clients = {k: [] for k in range(self.cfg["num_modalities"])}
        for c, p in zip(self.clients, pred):
            cluster_to_clients[int(p)].append(c.client_id)
        return cluster_to_clients

    def run(self):
        cluster_to_clients = self.cluster_clients()
        scheduler = FairRandomFullModalityScheduler(cluster_to_clients, seed=self.cfg["seed"])

        for round_idx in range(self.cfg["global_rounds"]):
            selected = scheduler.select()
            print(f"\n=== Global Round {round_idx} ===")
            print("selected clients:", selected)

            selected_clients = [self.clients_by_id[cid] for _, cid in sorted(selected.items())]
            for local_step in range(self.cfg["local_steps_per_round"]):
                print(f"-- Local Step {local_step}")

                payloads = []
                z_cache = {}
                for client in selected_clients:
                    x, y, _ = client.sample_batch()
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

        acc, macro_f1 = evaluate_paired_test(self.clients_by_modality, self.server, self.test_set, self.device)
        print("\n=== Paired multimodal test evaluation ===")
        print(f"test accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")
