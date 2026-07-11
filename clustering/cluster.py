import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score


class ISODATAClusterer:
    def __init__(self, max_iter=20, min_cluster_size=2, split_std_threshold=1.0, merge_distance_threshold=0.25, seed=0):
        self.max_iter = int(max_iter)
        self.min_cluster_size = int(min_cluster_size)
        self.split_std_threshold = float(split_std_threshold)
        self.merge_distance_threshold = float(merge_distance_threshold)
        self.seed = int(seed)

    def fit_predict(self, x, k):
        x = np.asarray(x, dtype=np.float32)
        target_k = int(k)
        labels = KMeans(n_clusters=target_k, random_state=self.seed, n_init=10).fit_predict(x)

        for _ in range(self.max_iter):
            centers = []
            for c in sorted(np.unique(labels)):
                members = x[labels == c]
                if len(members) >= self.min_cluster_size:
                    centers.append(members.mean(axis=0))
            if not centers:
                break
            centers = np.stack(centers)

            if len(centers) < target_k:
                largest = max(range(len(centers)), key=lambda c: int((labels == c).sum()))
                members = x[labels == largest]
                std = members.std(axis=0)
                axis = int(np.argmax(std))
                if float(std[axis]) > self.split_std_threshold or len(centers) < target_k:
                    delta = np.zeros(x.shape[1], dtype=np.float32)
                    delta[axis] = max(float(std[axis]), 1e-3) * 0.5
                    centers = np.concatenate([centers, centers[largest : largest + 1] + delta], axis=0)
            elif len(centers) > target_k:
                d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
                np.fill_diagonal(d, np.inf)
                a, b = np.unravel_index(np.argmin(d), d.shape)
                if d[a, b] < self.merge_distance_threshold or len(centers) > target_k:
                    keep = [i for i in range(len(centers)) if i not in {a, b}]
                    merged = ((centers[a] + centers[b]) / 2.0)[None, :]
                    centers = np.concatenate([centers[keep], merged], axis=0)

            dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

        if len(np.unique(labels)) != target_k:
            labels = KMeans(n_clusters=target_k, random_state=self.seed, n_init=10).fit_predict(x)
        return labels.astype(int)


def evaluate_clustering(gt, pred, k):
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    mapping = {}
    for c in range(k):
        members = gt[pred == c]
        if len(members) == 0:
            mapping[c] = -1
        else:
            vals, counts = np.unique(members, return_counts=True)
            mapping[c] = int(vals[np.argmax(counts)])
    mapped = np.array([mapping[int(p)] if mapping[int(p)] >= 0 else -1 for p in pred])

    valid = mapped >= 0
    acc = (mapped[valid] == gt[valid]).mean() if valid.any() else 0.0
    cm = confusion_matrix(gt, mapped, labels=list(range(k)))
    nmi = normalized_mutual_info_score(gt, pred)
    ari = adjusted_rand_score(gt, pred)
    return mapping, cm, float(acc), float(nmi), float(ari)


def standardize_features(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)


def run_kmeans(client_reps, known_k, seed=0):
    x = standardize_features(np.stack(client_reps))
    model = KMeans(n_clusters=int(known_k), random_state=int(seed), n_init=10)
    return model.fit_predict(x).astype(int)


def run_isodata(client_reps, known_k, seed=0, **kwargs):
    x = standardize_features(np.stack(client_reps))
    return ISODATAClusterer(seed=seed, **kwargs).fit_predict(x, int(known_k))
