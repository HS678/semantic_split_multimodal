import random
from collections import Counter, defaultdict


class BaseScheduler:
    def __init__(self, clients, clients_per_round, seed=0):
        self.clients = list(clients)
        self.clients_per_round = int(clients_per_round)
        self.rng = random.Random(int(seed))
        self.round_index = 0
        self.participation = Counter()

    def sample_round(self):
        raise NotImplementedError

    def _record(self, selected):
        self.round_index += 1
        for client in selected:
            self.participation[client.client_id] += 1
        return selected

    def metrics(self, selected):
        selected_clusters = {int(c.pred_cluster) for c in selected if c.pred_cluster is not None}
        all_clusters = {int(c.pred_cluster) for c in self.clients if c.pred_cluster is not None}
        counts = [self.participation[c.client_id] for c in self.clients]
        mean = sum(counts) / max(1, len(counts))
        fairness = 1.0
        if counts and sum(v * v for v in counts) > 0:
            fairness = (sum(counts) ** 2) / (len(counts) * sum(v * v for v in counts))
        return {
            "coverage": float(len(selected_clusters) / max(1, len(all_clusters))),
            "participation_fairness": float(fairness),
            "round": int(self.round_index),
        }


class RandomScheduler(BaseScheduler):
    def sample_round(self):
        k = min(self.clients_per_round, len(self.clients))
        return self._record(self.rng.sample(self.clients, k))


class RoundRobinScheduler(BaseScheduler):
    def __init__(self, clients, clients_per_round, seed=0):
        super().__init__(clients, clients_per_round, seed)
        self.order = list(self.clients)
        self.rng.shuffle(self.order)
        self.cursor = 0

    def sample_round(self):
        selected = []
        for _ in range(min(self.clients_per_round, len(self.order))):
            selected.append(self.order[self.cursor % len(self.order)])
            self.cursor += 1
        return self._record(selected)


class ProposedClusterCoverageScheduler(BaseScheduler):
    def sample_round(self):
        by_cluster = defaultdict(list)
        for client in self.clients:
            by_cluster[int(client.pred_cluster)].append(client)
        return self._record(_coverage_sample(by_cluster, self.clients, self.clients_per_round, self.rng))


def _coverage_sample(groups, all_clients, clients_per_round, rng):
    selected = []
    for key in sorted(groups):
        pool = list(groups[key])
        rng.shuffle(pool)
        selected.append(pool[0])
    remaining = [c for c in all_clients if c not in selected]
    rng.shuffle(remaining)
    need = max(0, int(clients_per_round) - len(selected))
    selected.extend(remaining[:need])
    if len(selected) > int(clients_per_round):
        rng.shuffle(selected)
        selected = selected[: int(clients_per_round)]
    return selected


def build_scheduler(name, clients, clients_per_round, seed=0):
    key = str(name).lower()
    if key == "random":
        return RandomScheduler(clients, clients_per_round, seed)
    if key == "round_robin":
        return RoundRobinScheduler(clients, clients_per_round, seed)
    if key in {"proposed_cluster_coverage", "cluster_coverage", "proposed"}:
        return ProposedClusterCoverageScheduler(clients, clients_per_round, seed)
    raise ValueError(f"Unsupported scheduler: {name}")
