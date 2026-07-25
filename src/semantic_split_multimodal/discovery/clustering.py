import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score


def relabel_contiguous(labels):
    labels = np.asarray(labels)
    mapping = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.array([mapping[v] for v in labels], dtype=int)


class ISODATAClusterer:
    """Small ISODATA-style clusterer for modality-aware client grouping."""

    def __init__(
        self,
        max_iter=20,
        min_cluster_size=2,
        split_std_threshold=1.0,
        merge_distance_threshold=0.25,
        initial_k=2,
        min_clusters=1,
        max_clusters=None,
        seed=0,
    ):
        self.max_iter = int(max_iter)
        self.min_cluster_size = int(min_cluster_size)
        self.split_std_threshold = float(split_std_threshold)
        self.merge_distance_threshold = float(merge_distance_threshold)
        self.initial_k = int(initial_k)
        self.min_clusters = int(min_clusters)
        self.max_clusters = max_clusters
        self.seed = int(seed)

    def _initial_labels(self, x, target_k):
        n = int(x.shape[0])
        if target_k is None:
            k = min(max(self.initial_k, self.min_clusters), n)
        else:
            k = min(max(int(target_k), 1), n)
        return KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit_predict(x)

    def _valid_centers(self, x, labels):
        centers = []
        for c in sorted(np.unique(labels)):
            members = x[labels == c]
            if len(members) >= self.min_cluster_size:
                centers.append(members.mean(axis=0))
        if not centers:
            return x.mean(axis=0, keepdims=True)
        return np.stack(centers)

    def _split_centers(self, x, labels, centers, target_k):
        max_clusters = self.max_clusters
        if max_clusters is None:
            max_clusters = int(target_k) if target_k is not None else max(
                self.min_clusters,
                min(int(x.shape[0]), self.initial_k * 3),
            )
        max_clusters = min(int(max_clusters), int(x.shape[0]))
        if len(centers) >= max_clusters:
            return centers

        new_centers = [center.copy() for center in centers]
        for c in range(len(centers)):
            if len(new_centers) >= max_clusters:
                break
            members = x[labels == c]
            if len(members) < max(2, self.min_cluster_size * 2):
                continue
            std = members.std(axis=0)
            axis = int(np.argmax(std))
            should_split = float(std[axis]) > self.split_std_threshold
            if target_k is not None:
                should_split = should_split or len(new_centers) < int(target_k)
            if not should_split:
                continue
            delta = np.zeros(x.shape[1], dtype=np.float32)
            delta[axis] = max(float(std[axis]), 1e-3) * 0.5
            new_centers[c] = centers[c] - delta
            new_centers.append(centers[c] + delta)
        return np.stack(new_centers)

    def _merge_centers(self, centers, target_k):
        min_clusters = int(target_k) if target_k is not None else self.min_clusters
        min_clusters = max(1, int(min_clusters))
        centers = centers.copy()
        while len(centers) > min_clusters:
            d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            a, b = np.unravel_index(np.argmin(d), d.shape)
            if target_k is None and float(d[a, b]) >= self.merge_distance_threshold:
                break
            keep = [i for i in range(len(centers)) if i not in {int(a), int(b)}]
            merged = ((centers[a] + centers[b]) / 2.0)[None, :]
            centers = np.concatenate([centers[keep], merged], axis=0)
        return centers

    def fit_predict(self, x, target_k=None):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"ISODATA expects a 2D feature matrix, got shape {x.shape}")
        if int(x.shape[0]) == 0:
            raise ValueError("ISODATA cannot cluster an empty feature matrix.")
        labels = self._initial_labels(x, target_k)

        for _ in range(self.max_iter):
            centers = self._valid_centers(x, labels)
            centers = self._split_centers(x, labels, centers, target_k)
            centers = self._merge_centers(centers, target_k)

            dist = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
            new_labels = dist.argmin(axis=1)
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

        labels = relabel_contiguous(labels)
        if target_k is not None and len(np.unique(labels)) != int(target_k):
            k = min(max(int(target_k), 1), int(x.shape[0]))
            labels = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit_predict(x)
            labels = relabel_contiguous(labels)
        return labels.astype(int)


def evaluate_clustering(gt, pred, k=None):
    gt = np.asarray(gt)
    pred = relabel_contiguous(pred)
    if k is None:
        k = int(max(len(np.unique(gt)), len(np.unique(pred))))
    else:
        k = int(k)
    mapping = {}
    for c in sorted(np.unique(pred)):
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
    if known_k is None:
        known_k = _estimate_k_by_silhouette(client_reps, seed)
    x = standardize_features(np.stack(client_reps))
    model = KMeans(n_clusters=int(known_k), random_state=int(seed), n_init=10)
    return model.fit_predict(x).astype(int)


def run_isodata(client_reps, known_k=None, seed=0, **kwargs):
    x = standardize_features(np.stack(client_reps))
    target_k = None if known_k is None else int(known_k)
    return ISODATAClusterer(seed=seed, **kwargs).fit_predict(x, target_k)


def run_hdbscan(client_reps, seed=0, **kwargs):
    try:
        import hdbscan
    except Exception as exc:
        raise ImportError("cluster.method: hdbscan requires the optional 'hdbscan' package.") from exc
    x = standardize_features(np.stack(client_reps))
    model = hdbscan.HDBSCAN(
        min_cluster_size=int(kwargs.get("min_cluster_size", 2)),
        min_samples=kwargs.get("min_samples"),
        cluster_selection_epsilon=float(kwargs.get("cluster_selection_epsilon", 0.0)),
    )
    labels = model.fit_predict(x)
    if np.any(labels < 0):
        next_label = int(labels.max()) + 1
        for i in np.where(labels < 0)[0]:
            labels[i] = next_label
            next_label += 1
    return relabel_contiguous(labels).astype(int)


def _estimate_k_by_silhouette(client_reps, seed=0):
    from sklearn.metrics import silhouette_score

    x = standardize_features(np.stack(client_reps))
    n = int(x.shape[0])
    if n <= 2:
        return 1
    best_k = 2
    best_score = -1.0
    for k in range(2, min(n, 8) + 1):
        labels = KMeans(n_clusters=k, random_state=int(seed), n_init=10).fit_predict(x)
        score = silhouette_score(x, labels)
        if score > best_score:
            best_k = k
            best_score = float(score)
    return best_k
