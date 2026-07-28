# Adaptive Modality Discovery

This branch adds an unknown-Q discovery path:

`PCA-denoised, conditional hard-partition shared-spherical-variance BIC-like adaptive ISODATA`

The estimator takes only client fingerprints. It does not accept hidden modality
labels, true modality count, dataset names, test labels, final task metrics, or
oracle evaluation mappings.

## Preprocessing

For every dataset, the same preprocessing is used:

1. Remove near-zero-variance fingerprint dimensions.
2. Standardize remaining dimensions.
3. Apply PCA denoising.
4. Select PCA dimension by cumulative explained variance, default `0.95`.
5. Cap PCA components by `num_clients - 1`.

Diagnostics record raw dimension, removed dimensions, PCA dimension, explained
variance ratios, and final clustering-space shape.

## BIC-Like Objective

The adaptive objective is a conditional hard-partition
shared-spherical-variance BIC-like objective in PCA space. Cluster assignments
are treated as conditionally given. For a partition with `k` clusters, `d` PCA
dimensions, `n` samples, and within-cluster SSE:

```text
sigma^2 = SSE / (n * d)
log L = -0.5 * n * d * (log(2*pi*sigma^2) + 1)
p = k*d + 1
score = log L - 0.5 * p * log(n)
```

The parameter count includes `k*d` centroid parameters and one shared variance.
Mixture weights are not modelled, so the likelihood and parameter count remain
consistent. This is not a complete Gaussian-mixture BIC. Higher score is better.
Split, merge, and candidate selection all use the same score.

## Split Rule

The estimator starts from one cluster. For each current cluster, it proposes a
two-cluster split with local k-means++ restarts derived from that run's seed. A split is
eligible for the candidate path when:

- both child clusters satisfy the unified minimum cluster size;
- the global BIC-like score improvement is positive after the complexity penalty;
- after at least one cluster already exists, the proposed split is
  contextually separated from the nearest external cluster.

Among eligible split proposals, the estimator accepts the one with the largest
global BIC-like score improvement into the candidate path. The final selected partition is
chosen later by label-free model selection over the path, which lets hierarchical
structures pass through a temporary silhouette dip.

The contextual split check is label-free. It prevents repeatedly carving an
already discovered compact cluster into small internal pieces unless the child
centers are at least as separated from each other as the parent is from nearby
external clusters. The unified threshold is `context_separation >= 1.1`, where
`context_separation` is the child-center distance divided by the distance from
the parent center to the nearest external cluster center. It does not use true
modality count, modality ID, modality name, labels, or dataset names.

`min_split_silhouette` is not a split-acceptance rule. It is a final
candidate-selection gate: among the accepted candidate path, only multi-cluster
candidates whose silhouette reaches this threshold can become the final
partition. If no multi-cluster candidate qualifies, the estimator keeps Q=1.

## Merge Rule

For each cluster pair, the estimator compares the current two-cluster model with
the merged single-cluster model using the same global BIC-like objective. It also
records the normalized center distance:

```text
D_ij = ||mu_i - mu_j|| / (r_i + r_j)
```

where `r_i` and `r_j` are robust within-cluster median radii. The merge is
accepted only if the merged partition improves the global BIC-like score.

## Stop Rule

Each iteration performs assignment/update, split proposal, optional best split,
merge proposal, optional best merge, and global objective checking. The run stops
when no operation is accepted, the global objective stops improving, a repeated
partition is detected, `q_max` is reached, `max_iter` is reached, or the
candidate path has not produced a better qualifying silhouette for the configured
patience window.

The selected partition is the candidate with the highest BIC-like score among
candidates whose silhouette reaches the global minimum threshold. Silhouette is
only the eligibility gate for final candidate selection.

`q_max` is a dataset-independent safety cap. The effective cap is also bounded
by `floor(num_clients / min_cluster_size)`, because every discovered cluster
must satisfy the minimum evidence rule. Reaching it sets
`boundary_saturation=true` and lowers selection confidence.

The current unknown-Q configs use `min_cluster_size=2` and `q_max=8`, so a
discoverable latent modality must be supported by at least two clients. A
single-client group is treated as an outlier or insufficient evidence, not as an
automatic new modality.

## Multi-Seed Consensus

Each configured seed runs a full adaptive discovery process and produces its own
`estimated_Q`. The final Q is selected by the Q frequency mode. Ties use the
smallest Q as a deterministic label-free rule. Within the selected Q, the medoid
run is chosen by highest average pairwise partition ARI against other runs.

The pairwise ARI compares unsupervised partitions only; it does not use hidden
modality labels.

## Post-Hoc Audit

After the final partition is locked, Stage2 may use `hidden_modality_id` only for
audit metrics:

- historical many-to-one majority `ACC`;
- one-to-one Hungarian `hungarian_ACC`;
- predicted-cluster by hidden-modality confusion matrix;
- cluster purity;
- true-modality split counts;
- predicted-cluster modality mix counts;
- NMI;
- ARI;
- `discovery_status`.

The audit status distinguishes:

- `discovery_success`;
- `correct_q_incorrect_partition`;
- `incorrect_q_over_or_under_clustering`;
- `discovery_failure`.

These audit values must not feed back into PCA, split, merge, Q selection, seed
selection, or parameter tuning.

`hidden_modality_name` is also forbidden in the adaptive estimator and
fingerprint/clustering path. It may exist in Stage1 metadata, but it must not be
used for PCA, split, merge, Q selection, seed selection, or parameter tuning.

## Known Limits

PCA denoising removes near-zero-variance nuisance dimensions before
standardization. High-variance random nuisance dimensions can still dominate the
small client-fingerprint sample regime and should be treated as a remaining
robustness risk rather than tuned with hidden modality labels.
