import random
from collections import Counter, defaultdict


class BalancedClusterRoundRobinScheduler:
    """Per-cluster random round-robin sampling without replacement."""

    def __init__(self, clients, clients_per_cluster_per_round, seed=0):
        self.clients = list(clients)
        self.clients_per_cluster_per_round = int(clients_per_cluster_per_round)
        if self.clients_per_cluster_per_round <= 0:
            raise ValueError("clients_per_cluster_per_round must be positive.")
        self.rng = random.Random(int(seed))
        self.round_index = 0
        self.participation = Counter()
        self.by_cluster = defaultdict(list)
        for client in self.clients:
            self.by_cluster[int(client.pred_cluster)].append(client)
        if not self.by_cluster:
            raise ValueError("Balanced scheduler requires at least one predicted cluster.")
        for cluster_id, group in self.by_cluster.items():
            if len(group) < self.clients_per_cluster_per_round:
                raise ValueError(
                    "clients_per_cluster_per_round cannot exceed the number of clients "
                    f"in pred_cluster {cluster_id}: r={self.clients_per_cluster_per_round}, "
                    f"num_clients={len(group)}"
                )
        self.cluster_ids = sorted(self.by_cluster)
        self.pools = {
            cluster_id: self._new_pool(cluster_id, exclude=set())
            for cluster_id in self.cluster_ids
        }

    @property
    def clients_per_round(self):
        return self.clients_per_cluster_per_round * len(self.cluster_ids)

    def _new_pool(self, cluster_id, exclude):
        pool = [client for client in self.by_cluster[cluster_id] if client.client_id not in exclude]
        self.rng.shuffle(pool)
        return pool

    def _sample_cluster(self, cluster_id):
        selected = []
        selected_ids = set()
        while len(selected) < self.clients_per_cluster_per_round:
            pool = self.pools[cluster_id]
            if not pool:
                self.pools[cluster_id] = self._new_pool(cluster_id, selected_ids)
                pool = self.pools[cluster_id]
            client = pool.pop()
            if client.client_id in selected_ids:
                continue
            selected.append(client)
            selected_ids.add(client.client_id)
        return selected

    def sample_round(self):
        selected = []
        for cluster_id in self.cluster_ids:
            selected.extend(self._sample_cluster(cluster_id))
        self.rng.shuffle(selected)
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
            "clients_per_cluster_per_round": int(self.clients_per_cluster_per_round),
            "per_cluster_selected": {
                str(cluster_id): int(per_cluster_selected.get(cluster_id, 0))
                for cluster_id in self.cluster_ids
            },
        }


def build_scheduler(name, clients, clients_per_cluster_per_round, seed=0):
    key = str(name).lower()
    if key in {"balanced_cluster_round_robin", "balanced"}:
        return BalancedClusterRoundRobinScheduler(clients, clients_per_cluster_per_round, seed)
    raise ValueError("Unsupported scheduler. Use 'balanced_cluster_round_robin'.")
