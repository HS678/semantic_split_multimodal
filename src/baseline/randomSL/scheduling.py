"""Random client scheduler for the randomSL baseline."""

import random
from collections import Counter, defaultdict


class RandomScheduler:
    """Uniform random sampling without replacement; no cluster coverage guarantee.

    Sampling does not read any modality information. ``pred_cluster`` is only
    read in :meth:`metrics` for audit/reporting, consistent with the MSL
    mainline rule that all training-time modality information must come from
    the predicted clusters.
    """

    def __init__(self, clients, clients_per_round, seed=0):
        self.clients = list(clients)
        self.clients_per_round = int(clients_per_round)
        if self.clients_per_round <= 0:
            raise ValueError("clients_per_round must be positive.")
        self.rng = random.Random(int(seed))
        self.round_index = 0
        self.participation = Counter()
        self.by_cluster = defaultdict(list)
        for client in self.clients:
            self.by_cluster[int(client.pred_cluster)].append(client)
        if not self.by_cluster:
            raise ValueError("RandomScheduler requires at least one client.")
        self.cluster_ids = sorted(self.by_cluster)

    def sample_round(self):
        k = min(self.clients_per_round, len(self.clients))
        selected = self.rng.sample(self.clients, k)
        self.round_index += 1
        for client in selected:
            self.participation[client.client_id] += 1
        return selected

    def metrics(self, selected):
        selected_clusters = {int(client.pred_cluster) for client in selected}
        counts = [self.participation[client.client_id] for client in self.clients]
        fairness = 1.0
        if counts and sum(value * value for value in counts) > 0:
            fairness = (sum(counts) ** 2) / (len(counts) * sum(value * value for value in counts))
        per_cluster_selected = Counter(int(client.pred_cluster) for client in selected)
        return {
            "coverage": float(len(selected_clusters) / max(1, len(self.cluster_ids))),
            "participation_fairness": float(fairness),
            "round": int(self.round_index),
            "clients_per_round": int(self.clients_per_round),
            "per_cluster_selected": {
                str(cluster_id): int(per_cluster_selected.get(cluster_id, 0))
                for cluster_id in self.cluster_ids
            },
        }
