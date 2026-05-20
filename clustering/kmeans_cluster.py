import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score


class ISODATAClusterer:
    def fit_predict(self, x, k):
        raise NotImplementedError("ISODATA is a v1 stub.")


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
    mapped = np.array([mapping[p] if mapping[p] >= 0 else -1 for p in pred])

    valid = mapped >= 0
    acc = (mapped[valid] == gt[valid]).mean() if valid.any() else 0.0
    cm = confusion_matrix(gt, mapped, labels=list(range(k)))
    nmi = normalized_mutual_info_score(gt, pred)
    ari = adjusted_rand_score(gt, pred)
    return mapping, cm, float(acc), float(nmi), float(ari)


def run_kmeans(client_reps, known_k, seed=0):
    x = np.stack(client_reps)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std(axis=0, keepdims=True) + 1e-6)
    model = KMeans(n_clusters=known_k, random_state=seed, n_init=10)
    pred = model.fit_predict(x)
    return pred
