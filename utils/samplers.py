from collections import Counter
import random


class ClassBalancedBatchSampler:
    """Best-effort class-balanced sampler over existing labels in a client."""

    def __init__(self, labels, batch_size, rng_seed=0):
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        self.rng = random.Random(rng_seed)
        self.label_to_indices = {}
        for idx, y in enumerate(self.labels):
            self.label_to_indices.setdefault(int(y), []).append(idx)
        self.classes = sorted(self.label_to_indices.keys())

    def sample_indices(self):
        if not self.classes:
            return []
        per_class = max(1, self.batch_size // len(self.classes))
        picked = []
        for c in self.classes:
            pool = self.label_to_indices[c]
            if len(pool) >= per_class:
                picked.extend(self.rng.sample(pool, per_class))
            elif pool:
                picked.extend(self.rng.choices(pool, k=per_class))
        while len(picked) < self.batch_size:
            c = self.rng.choice(self.classes)
            picked.append(self.rng.choice(self.label_to_indices[c]))
        self.rng.shuffle(picked)
        return picked[: self.batch_size]

    def describe_batch(self, batch_labels):
        return dict(Counter(int(x) for x in batch_labels))
