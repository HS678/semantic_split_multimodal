# Adaptive ISODATA、KMeans 和聚类可行性相关 discovery 算法。
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def relabel_contiguous(labels):
    labels = np.asarray(labels)
    mapping = {old: new for new, old in enumerate(sorted(np.unique(labels)))}
    return np.array([mapping[v] for v in labels], dtype=int)


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


def adaptive_isodata(client_reps, seed=0, **kwargs):
    estimator = AdaptiveISODATAEstimator(seed=seed, **kwargs)
    labels = estimator.fit_predict(client_reps)
    return labels, estimator.diagnostics_


class AdaptiveISODATAEstimator:
    """PCA-denoised, BIC-like-score-guided adaptive ISODATA.

    The unsupervised score is a conditional hard-partition
    shared-spherical-variance BIC-like objective in PCA space. Cluster
    assignments are treated as conditionally given. For a partition with k
    clusters, d PCA dimensions, n samples, and within-cluster SSE, the
    conditional log-likelihood at the MLE variance is:

        log L = -0.5 * n * d * (log(2*pi*sigma^2) + 1)

    with sigma^2 = SSE / (n*d). The parameter count is k*d centroid parameters
    plus one shared variance: p = k*d + 1. Mixture weights are not modelled.
    We maximize score = log L - 0.5 * p * log(n). The same higher-is-better
    score is used by split, merge, and candidate selection.
    """

    def __init__(
        self,
        seeds=(11, 23, 37, 53, 71),
        max_iter=20,
        pca_variance=0.95,
        variance_epsilon=1e-8,
        q_max=None,
        min_cluster_size=None,
        min_cluster_size_fraction=None,
        bic_improvement_min=0.0,
        min_split_silhouette=0.25,
        silhouette_patience=2,
        stability_min_ari=0.75,
        split_kmeans_restarts=50,
        seed=0,
    ):
        self.seeds = [int(v) for v in seeds] if seeds is not None else [int(seed)]
        if not self.seeds:
            self.seeds = [int(seed)]
        self.max_iter = int(max_iter)
        self.pca_variance = float(pca_variance)
        self.variance_epsilon = float(variance_epsilon)
        self.q_max = None if q_max is None else int(q_max)
        self.min_cluster_size = None if min_cluster_size is None else int(min_cluster_size)
        self.min_cluster_size_fraction = None if min_cluster_size_fraction is None else float(min_cluster_size_fraction)
        self.bic_improvement_min = float(bic_improvement_min)
        self.min_split_silhouette = float(min_split_silhouette)
        self.silhouette_patience = int(silhouette_patience)
        self.stability_min_ari = float(stability_min_ari)
        self.split_kmeans_restarts = int(split_kmeans_restarts)
        self.seed = int(seed)
        self.diagnostics_ = {}

    def fit_predict(self, client_reps):
        raw = np.asarray(np.stack(client_reps), dtype=np.float32)
        x, preprocessing = _adaptive_preprocess(
            raw,
            pca_variance=self.pca_variance,
            variance_epsilon=self.variance_epsilon,
        )
        n = int(x.shape[0])
        min_size = self.min_cluster_size
        if min_size is None:
            if self.min_cluster_size_fraction is None:
                min_size = 2
            else:
                min_size = max(2, int(np.ceil(n * self.min_cluster_size_fraction)))
        min_size = max(1, min(int(min_size), n))
        q_max = self.q_max if self.q_max is not None else _default_q_max(n)
        q_max = max(1, min(int(q_max), n // max(1, min_size)))

        runs = []
        for seed in self.seeds:
            runs.append(
                _run_adaptive_isodata_single(
                    x,
                    seed=int(seed),
                    max_iter=self.max_iter,
                    q_max=q_max,
                    min_cluster_size=min_size,
                    bic_improvement_min=self.bic_improvement_min,
                    min_split_silhouette=self.min_split_silhouette,
                    silhouette_patience=self.silhouette_patience,
                    split_kmeans_restarts=self.split_kmeans_restarts,
                )
            )

        labels, selection = _select_consensus_run(runs)
        self.diagnostics_ = _build_adaptive_diagnostics(
            x, labels, runs, selection, preprocessing, q_max, min_size, self
        )
        return labels.astype(int)


def _adaptive_preprocess(raw, pca_variance=0.95, variance_epsilon=1e-8):
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Adaptive ISODATA expects a 2D feature matrix, got shape {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError("Adaptive ISODATA received non-finite client representations.")
    n, d = raw.shape
    if n == 0:
        raise ValueError("Adaptive ISODATA cannot cluster an empty feature matrix.")
    std = raw.std(axis=0)
    keep = std > float(variance_epsilon)
    if not np.any(keep):
        keep[np.argmax(std)] = True
    filtered = raw[:, keep]
    standardized = standardize_features(filtered)
    max_components = max(1, min(int(standardized.shape[1]), int(n) - 1 if n > 1 else 1))
    if max_components == 1:
        space = standardized[:, :1]
        explained = [1.0]
    else:
        pca = PCA(n_components=max_components, svd_solver="full")
        transformed = pca.fit_transform(standardized)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        dim = int(np.searchsorted(cumulative, float(pca_variance), side="left") + 1)
        dim = max(1, min(dim, max_components))
        space = transformed[:, :dim]
        explained = pca.explained_variance_ratio_[:dim].astype(float).tolist()
    return space.astype(np.float32), {
        "raw_dim": int(d),
        "removed_near_zero_variance_dims": int(d - int(np.sum(keep))),
        "pca_dim": int(space.shape[1]),
        "explained_variance_ratio": explained,
        "explained_variance_sum": float(np.sum(explained)),
        "clustering_space_shape": [int(v) for v in space.shape],
        "pca_variance_target": float(pca_variance),
        "variance_epsilon": float(variance_epsilon),
    }


def _default_q_max(n):
    return max(1, min(int(n), int(np.ceil(np.sqrt(max(1, n))) + 2)))


def _run_adaptive_isodata_single(
    x,
    seed,
    max_iter,
    q_max,
    min_cluster_size,
    bic_improvement_min,
    min_split_silhouette,
    silhouette_patience,
    split_kmeans_restarts,
):
    labels = np.zeros(int(x.shape[0]), dtype=int)
    objective = _partition_bic(x, labels)
    candidates = [_candidate_snapshot(x, labels, "initial")]
    best_candidate_score = float("-inf")
    no_selection_improvement = 0
    split_history = []
    merge_history = []
    seen = {_partition_key(labels)}
    convergence_reason = "max_iter"
    boundary_saturation = False

    for iteration in range(1, int(max_iter) + 1):
        labels = _reassign_to_centers(x, labels)
        labels = relabel_contiguous(labels)
        current = _partition_bic(x, labels)

        split = _best_split_proposal(
            x,
            labels,
            seed=seed + iteration * 1009,
            q_max=q_max,
            min_cluster_size=min_cluster_size,
            bic_improvement_min=bic_improvement_min,
            min_split_silhouette=min_split_silhouette,
            split_kmeans_restarts=split_kmeans_restarts,
        )
        changed = False
        if split is not None and split.get("eligible") and split["new_global_bic"] > current + bic_improvement_min:
            labels = _reassign_to_centers(x, relabel_contiguous(split["labels"]))
            labels = relabel_contiguous(labels)
            split["accepted"] = True
            split_history.append(_proposal_without_labels(split))
            current = _partition_bic(x, labels)
            changed = True
            candidates.append(_candidate_snapshot(x, labels, "split"))
        elif split is not None:
            split["accepted"] = False
            split_history.append(_proposal_without_labels(split))

        merge = _best_merge_proposal(x, labels, bic_improvement_min)
        if merge is not None and merge["new_global_bic"] > current + bic_improvement_min:
            labels = _reassign_to_centers(x, relabel_contiguous(merge["labels"]))
            labels = relabel_contiguous(labels)
            merge["accepted"] = True
            merge_history.append(_proposal_without_labels(merge))
            changed = True
            candidates.append(_candidate_snapshot(x, labels, "merge"))
        elif merge is not None:
            merge["accepted"] = False
            merge_history.append(_proposal_without_labels(merge))

        key = _partition_key(labels)
        if key in seen:
            convergence_reason = "repeated_partition"
            break
        seen.add(key)

        new_objective = _partition_bic(x, labels)
        if new_objective <= objective + bic_improvement_min:
            convergence_reason = "global_objective_not_improved"
            break
        objective = new_objective
        if len(np.unique(labels)) >= int(q_max):
            boundary_saturation = True
            convergence_reason = "q_max_reached"
            break
        if not changed:
            convergence_reason = "no_acceptable_split_or_merge"
            break
        current_candidate = candidates[-1]
        selection_score = current_candidate["selection_score"]
        if selection_score is not None and selection_score > best_candidate_score:
            best_candidate_score = float(selection_score)
            no_selection_improvement = 0
        else:
            no_selection_improvement += 1
        if no_selection_improvement >= int(silhouette_patience):
            convergence_reason = "unsupervised_selection_patience"
            break

    selected_candidate = _select_best_candidate(candidates, min_split_silhouette)
    labels = relabel_contiguous(selected_candidate["labels"])
    return {
        "seed": int(seed),
        "labels": labels.astype(int),
        "estimated_Q": int(len(np.unique(labels))),
        "cluster_sizes": _cluster_sizes(labels),
        "objective": float(_partition_bic(x, labels)),
        "split_history": split_history,
        "merge_history": merge_history,
        "candidate_history": [_candidate_without_labels(item) for item in candidates],
        "selected_candidate": _candidate_without_labels(selected_candidate),
        "convergence_reason": convergence_reason,
        "boundary_saturation": bool(boundary_saturation or len(np.unique(labels)) >= int(q_max)),
    }


def _partition_bic(x, labels):
    labels = relabel_contiguous(labels)
    n, d = int(x.shape[0]), int(x.shape[1])
    k = int(len(np.unique(labels)))
    sse = _partition_sse(x, labels)
    sigma2 = max(float(sse) / max(1, n * d), 1e-9)
    log_likelihood = -0.5 * n * d * (np.log(2.0 * np.pi * sigma2) + 1.0)
    params = k * d + 1
    return float(log_likelihood - 0.5 * params * np.log(max(2, n)))


def _partition_sse(x, labels):
    total = 0.0
    for cluster_id in np.unique(labels):
        members = x[labels == cluster_id]
        if len(members) == 0:
            continue
        center = members.mean(axis=0, keepdims=True)
        total += float(((members - center) ** 2).sum())
    return total


def _best_split_proposal(
    x,
    labels,
    seed,
    q_max,
    min_cluster_size,
    bic_improvement_min,
    min_split_silhouette,
    split_kmeans_restarts,
):
    if len(np.unique(labels)) >= int(q_max):
        return None
    current_bic = _partition_bic(x, labels)
    current_silhouette = _silhouette_or_none(x, labels)
    best = None
    for cluster_id in sorted(np.unique(labels)):
        idx = np.where(labels == cluster_id)[0]
        if len(idx) < int(min_cluster_size) * 2:
            continue
        local = x[idx]
        best_local = None
        rng = np.random.default_rng(int(seed))
        restart_seeds = rng.integers(0, np.iinfo(np.int32).max, size=max(1, int(split_kmeans_restarts)))
        for offset in range(max(1, int(split_kmeans_restarts))):
            restart_seed = int(restart_seeds[offset])
            km = KMeans(n_clusters=2, init="k-means++", random_state=restart_seed, n_init=1)
            local_labels = km.fit_predict(local)
            sizes = np.bincount(local_labels, minlength=2)
            if np.any(sizes < int(min_cluster_size)):
                continue
            local_bic_before = _partition_bic(local, np.zeros(len(local), dtype=int))
            local_bic_after = _partition_bic(local, local_labels)
            improvement = float(local_bic_after - local_bic_before)
            if best_local is None or improvement > best_local["local_bic_improvement"]:
                best_local = {
                    "local_labels": local_labels.astype(int),
                    "local_bic_improvement": improvement,
                    "child_sizes": [int(v) for v in sizes.tolist()],
                    "restart_seed": restart_seed,
                }
        if best_local is None:
            continue
        new_labels = labels.copy()
        next_id = int(labels.max()) + 1
        new_labels[idx[best_local["local_labels"] == 1]] = next_id
        new_labels = relabel_contiguous(new_labels)
        new_bic = _partition_bic(x, new_labels)
        improvement = float(new_bic - current_bic)
        new_silhouette = _silhouette_or_none(x, new_labels)
        context_separation = _split_context_separation(x, labels, new_labels, int(cluster_id))
        reason = "bic_improved"
        if improvement <= bic_improvement_min:
            reason = "bic_penalty_not_paid"
        proposal = {
            "type": "split",
            "cluster_id": int(cluster_id),
            "child_sizes": best_local["child_sizes"],
            "restart_seed": int(best_local["restart_seed"]),
            "local_bic_improvement": best_local["local_bic_improvement"],
            "score_improvement": improvement,
            "old_silhouette": None if current_silhouette is None else float(current_silhouette),
            "new_silhouette": None if new_silhouette is None else float(new_silhouette),
            "min_split_silhouette": float(min_split_silhouette),
            "context_separation": None if context_separation is None else float(context_separation),
            "context_role": "diagnostic_only",
            "old_global_bic": float(current_bic),
            "new_global_bic": float(new_bic),
            "reason": reason,
            "eligible": bool(improvement > bic_improvement_min),
            "labels": new_labels,
        }
        if best is None or _proposal_rank(proposal) > _proposal_rank(best):
            best = proposal
    return best


def _best_merge_proposal(x, labels, bic_improvement_min):
    labels = relabel_contiguous(labels)
    clusters = sorted(np.unique(labels))
    if len(clusters) <= 1:
        return None
    current_bic = _partition_bic(x, labels)
    best = None
    for i, first in enumerate(clusters):
        for second in clusters[i + 1 :]:
            merged = labels.copy()
            merged[merged == second] = first
            merged = relabel_contiguous(merged)
            new_bic = _partition_bic(x, merged)
            improvement = float(new_bic - current_bic)
            radius_i = _cluster_radius(x[labels == first])
            radius_j = _cluster_radius(x[labels == second])
            center_i = x[labels == first].mean(axis=0)
            center_j = x[labels == second].mean(axis=0)
            normalized_distance = float(np.linalg.norm(center_i - center_j) / max(radius_i + radius_j, 1e-9))
            proposal = {
                "type": "merge",
                "cluster_pair": [int(first), int(second)],
                "score_improvement": improvement,
                "old_global_bic": float(current_bic),
                "new_global_bic": float(new_bic),
                "normalized_center_distance": normalized_distance,
                "reason": "bic_improved" if improvement > bic_improvement_min else "merged_model_not_better",
                "labels": merged,
            }
            if best is None or proposal["score_improvement"] > best["score_improvement"]:
                best = proposal
    return best


def _proposal_rank(proposal):
    return (1 if proposal.get("eligible", True) else 0, float(proposal["score_improvement"]))


def _split_context_separation(x, labels, new_labels, split_cluster_id):
    labels = relabel_contiguous(labels)
    new_labels = relabel_contiguous(new_labels)
    clusters = sorted(np.unique(labels))
    if len(clusters) <= 1:
        return None
    idx = np.where(labels == int(split_cluster_id))[0]
    child_labels = relabel_contiguous(new_labels[idx])
    child_values = sorted(np.unique(child_labels))
    if len(child_values) != 2:
        return None
    first = x[idx][child_labels == child_values[0]]
    second = x[idx][child_labels == child_values[1]]
    if len(first) == 0 or len(second) == 0:
        return 0.0
    child_distance = float(np.linalg.norm(first.mean(axis=0) - second.mean(axis=0)))
    parent_center = x[idx].mean(axis=0)
    external_distances = [
        float(np.linalg.norm(parent_center - x[labels == cluster_id].mean(axis=0)))
        for cluster_id in clusters
        if int(cluster_id) != int(split_cluster_id)
    ]
    if not external_distances:
        return None
    return float(child_distance / max(min(external_distances), 1e-9))


def _candidate_snapshot(x, labels, operation):
    labels = relabel_contiguous(labels)
    scores = _unsupervised_scores(x, labels)
    return {
        "operation": operation,
        "labels": labels.astype(int),
        "estimated_Q": int(len(np.unique(labels))),
        "cluster_sizes": _cluster_sizes(labels),
        "objective": float(_partition_bic(x, labels)),
        "silhouette": scores["silhouette"],
        "DBI": scores["DBI"],
        "CH": scores["CH"],
        "selection_score": scores["silhouette"],
    }


def _select_best_candidate(candidates, min_split_silhouette):
    eligible = [
        item
        for item in candidates
        if item["estimated_Q"] > 1
        and item["selection_score"] is not None
        and item["selection_score"] >= float(min_split_silhouette)
    ]
    if not eligible:
        return candidates[0]
    return max(eligible, key=lambda item: (float(item["objective"]), float(item["selection_score"]), -item["estimated_Q"]))


def _candidate_without_labels(candidate):
    return {key: value for key, value in candidate.items() if key != "labels"}


def _cluster_radius(members):
    if len(members) <= 1:
        return 0.0
    center = members.mean(axis=0, keepdims=True)
    distances = np.linalg.norm(members - center, axis=1)
    return float(np.median(distances))


def _reassign_to_centers(x, labels):
    labels = relabel_contiguous(labels)
    centers = np.stack([x[labels == cluster_id].mean(axis=0) for cluster_id in sorted(np.unique(labels))])
    distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
    return distances.argmin(axis=1).astype(int)


def _partition_key(labels):
    labels = relabel_contiguous(labels)
    return tuple(int(v) for v in labels.tolist())


def _proposal_without_labels(proposal):
    return {key: value for key, value in proposal.items() if key != "labels"}


def _cluster_sizes(labels):
    counts = Counter(int(v) for v in labels)
    return {str(k): int(counts[k]) for k in sorted(counts)}


def _select_consensus_run(runs):
    q_counts = Counter(int(run["estimated_Q"]) for run in runs)
    max_count = max(q_counts.values())
    selected_q = min(q for q, count in q_counts.items() if count == max_count)
    candidate_indices = [idx for idx, run in enumerate(runs) if int(run["estimated_Q"]) == selected_q]
    agreement = _pairwise_run_ari(runs)
    best_idx = candidate_indices[0]
    best_agreement = -np.inf
    for idx in candidate_indices:
        values = [agreement[f"{min(idx, j)}-{max(idx, j)}"] for j in range(len(runs)) if j != idx]
        avg = float(np.mean(values)) if values else 1.0
        if avg > best_agreement or (avg == best_agreement and runs[idx]["objective"] > runs[best_idx]["objective"]):
            best_idx = idx
            best_agreement = avg
    return runs[best_idx]["labels"].copy(), {
        "selected_run_index": int(best_idx),
        "selected_seed": int(runs[best_idx]["seed"]),
        "selected_Q": int(selected_q),
        "q_frequency": {str(k): int(v) for k, v in sorted(q_counts.items())},
        "mean_agreement_for_selected": float(best_agreement if np.isfinite(best_agreement) else 1.0),
    }


def _pairwise_run_ari(runs):
    out = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            out[f"{i}-{j}"] = float(adjusted_rand_score(runs[i]["labels"], runs[j]["labels"]))
    return out


def _build_adaptive_diagnostics(x, labels, runs, selection, preprocessing, q_max, min_size, estimator):
    pairwise = _pairwise_run_ari(runs)
    pairwise_values = list(pairwise.values())
    q_freq = selection["q_frequency"]
    q_stability = max(q_freq.values()) / max(1, len(runs))
    assignment_stability = float(np.mean(pairwise_values)) if pairwise_values else 1.0
    labels = relabel_contiguous(labels)
    scores = _unsupervised_scores(x, labels)
    boundary = bool(len(np.unique(labels)) >= int(q_max) or any(run["boundary_saturation"] for run in runs))
    confidence = "high"
    if boundary or q_stability < 0.8 or assignment_stability < estimator.stability_min_ari:
        confidence = "low"
    elif q_stability < 1.0:
        confidence = "medium"
    selected = runs[selection["selected_run_index"]]
    return {
        "q_source": "adaptive_isodata",
        "preprocessing": preprocessing,
        "algorithm_config": {
            "objective_name": "conditional hard-partition shared-spherical-variance BIC-like objective",
            "seeds": [int(v) for v in estimator.seeds],
            "max_iter": int(estimator.max_iter),
            "pca_variance": float(estimator.pca_variance),
            "variance_epsilon": float(estimator.variance_epsilon),
            "q_max": int(q_max),
            "min_cluster_size": int(min_size),
            "min_cluster_size_fraction": (
                None if estimator.min_cluster_size_fraction is None else float(estimator.min_cluster_size_fraction)
            ),
            "bic_improvement_min": float(estimator.bic_improvement_min),
            "min_split_silhouette": float(estimator.min_split_silhouette),
            "silhouette_patience": int(estimator.silhouette_patience),
            "stability_min_ari": float(estimator.stability_min_ari),
            "split_kmeans_restarts": int(estimator.split_kmeans_restarts),
        },
        "estimated_Q": int(len(np.unique(labels))),
        "cluster_sizes": _cluster_sizes(labels),
        "per_seed_estimated_Q": [int(run["estimated_Q"]) for run in runs],
        "per_seed_cluster_sizes": [run["cluster_sizes"] for run in runs],
        "per_seed_objective": [float(run["objective"]) for run in runs],
        "pairwise_partition_ari": pairwise,
        "q_frequency": q_freq,
        "q_stability": float(q_stability),
        "assignment_stability": float(assignment_stability),
        "selection": selection,
        "selection_confidence": confidence,
        "boundary_saturation": boundary,
        "minimum_cluster_size": int(min_size),
        "small_cluster_present": bool(any(int(v) < int(min_size) for v in Counter(labels).values())),
        "silhouette": scores["silhouette"],
        "DBI": scores["DBI"],
        "CH": scores["CH"],
        "split_history": selected["split_history"],
        "merge_history": selected["merge_history"],
        "split_count": int(sum(1 for item in selected["split_history"] if item.get("accepted"))),
        "merge_count": int(sum(1 for item in selected["merge_history"] if item.get("accepted"))),
        "convergence_reason": selected["convergence_reason"],
        "runs": [{key: value for key, value in run.items() if key != "labels"} for run in runs],
    }


def _unsupervised_scores(x, labels):
    labels = relabel_contiguous(labels)
    if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
        return {"silhouette": None, "DBI": None, "CH": None}
    return {
        "silhouette": float(silhouette_score(x, labels)),
        "DBI": float(davies_bouldin_score(x, labels)),
        "CH": float(calinski_harabasz_score(x, labels)),
    }


def _silhouette_or_none(x, labels):
    labels = relabel_contiguous(labels)
    if len(np.unique(labels)) <= 1 or len(np.unique(labels)) >= len(labels):
        return None
    return float(silhouette_score(x, labels))


def run_kmeans(client_reps, known_k, seed=0):
    if known_k is None:
        known_k = _estimate_k_by_silhouette(client_reps, seed)
    x = standardize_features(np.stack(client_reps))
    model = KMeans(n_clusters=int(known_k), random_state=int(seed), n_init=10)
    return model.fit_predict(x).astype(int)


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


import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def encoder_fingerprint(encoder, samples, lengths=None, batch_size=64, max_batches=4, device="cpu"):
    dataset = TensorDataset(samples, lengths) if lengths is not None else TensorDataset(samples)
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    chunks = []
    encoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            xb = batch[0].to(device)
            length_batch = batch[1].to(device) if len(batch) > 1 else None
            encoded = encoder(xb) if length_batch is None else encoder(xb, length_batch)
            chunks.append(encoded.detach().cpu())
    if not chunks:
        raise RuntimeError("Cannot extract fingerprint from an empty client dataset.")
    z = torch.cat(chunks, dim=0)
    return torch.cat([z.mean(dim=0), z.std(dim=0)], dim=0).numpy().astype(np.float32)


def signal_stat_fingerprint(samples):
    x = samples.detach().float()
    if x.dim() == 1:
        x = x.reshape(-1, 1, 1)
    elif x.dim() == 2:
        x = x.reshape(x.shape[0], 1, x.shape[1])
    reduce_dims = tuple(i for i in range(x.dim()) if i != 1)
    stats = [
        x.mean(dim=reduce_dims),
        x.std(dim=reduce_dims),
        x.abs().mean(dim=reduce_dims),
        x.amax(dim=reduce_dims),
        x.amin(dim=reduce_dims),
    ]
    return torch.cat(stats).numpy().astype(np.float32)


def pad_feature_rows(rows):
    max_len = max(len(row) for row in rows)
    out = np.zeros((len(rows), max_len), dtype=np.float32)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def build_fingerprints(clients, encoders, cfg, device):
    fp_cfg = cfg.get("fingerprint", {})
    source = str(fp_cfg.get("type", fp_cfg.get("source", "hybrid"))).lower()
    batch_size = int(fp_cfg.get("batch_size", cfg.get("batch_size", 64)))
    max_batches = fp_cfg.get("max_batches", 4)
    rows = []
    for client in clients:
        parts = []
        if source in {"encoder", "hybrid"}:
            parts.append(
                encoder_fingerprint(
                    encoders[client.client_id],
                    client.samples,
                    client.sequence_lengths,
                    batch_size,
                    max_batches,
                    device,
                )
            )
        if source in {"signal", "signal_stats", "hybrid"}:
            parts.append(signal_stat_fingerprint(client.samples))
        if not parts:
            raise ValueError("fingerprint.type must be 'encoder', 'signal', or 'hybrid'.")
        rows.append(np.concatenate(parts).astype(np.float32))
    return pad_feature_rows(rows)
