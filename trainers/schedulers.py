import random


class FairRandomFullModalityScheduler:
    def __init__(self, cluster_to_clients, seed=0):
        self.cluster_to_clients = cluster_to_clients
        self.rng = random.Random(seed)
        self.pools = {k: [] for k in cluster_to_clients}

    def _refresh(self, cluster_id):
        self.pools[cluster_id] = list(self.cluster_to_clients[cluster_id])
        self.rng.shuffle(self.pools[cluster_id])

    def select(self):
        selected = {}
        for cluster_id in sorted(self.cluster_to_clients.keys()):
            if not self.pools[cluster_id]:
                self._refresh(cluster_id)
            selected[cluster_id] = self.pools[cluster_id].pop()
        return selected


class PairedFullModalityScheduler:
    def select(self):
        raise NotImplementedError("PairedFullModalityScheduler is a v1 stub.")


class GlobalRandomScheduler:
    def select(self):
        raise NotImplementedError("GlobalRandomScheduler is a v1 stub.")
