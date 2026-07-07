# 随机创建模拟数据
import numpy as np
import torch


def make_synthetic_paired_dataset(num_samples, num_modalities, num_classes, input_dim, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=num_samples)
    class_centers = rng.normal(0, 1, size=(num_modalities, num_classes, input_dim))
    modality_bias = rng.normal(0, 0.5, size=(num_modalities, input_dim))

    xs = []
    for m in range(num_modalities):
        noise = rng.normal(0, 1.0, size=(num_samples, input_dim))
        x = class_centers[m, labels] + modality_bias[m] + noise
        xs.append(x.astype(np.float32))

    return {
        "modalities": [torch.tensor(x) for x in xs],
        "labels": torch.tensor(labels, dtype=torch.long),
        "modality_input_dims": [int(input_dim) for _ in range(num_modalities)],
    }


def split_train_test(dataset, train_ratio):
    n = len(dataset["labels"])
    train_n = int(n * train_ratio)
    out = {}
    modality_input_dims = dataset.get(
        "modality_input_dims",
        [int(dataset["modalities"][0].shape[1]) for _ in range(len(dataset["modalities"]))],
    )
    out["train"] = {
        "modalities": [x[:train_n] for x in dataset["modalities"]],
        "labels": dataset["labels"][:train_n],
        "modality_input_dims": modality_input_dims,
    }
    out["test"] = {
        "modalities": [x[train_n:] for x in dataset["modalities"]],
        "labels": dataset["labels"][train_n:],
        "modality_input_dims": modality_input_dims,
    }
    return out


def _dirichlet_weights(num_clients, alpha, rng):
    w = rng.dirichlet(np.ones(num_clients) * alpha)
    w = w / np.sum(w)
    return w


def partition_modality_to_clients(
    x_mod,
    y,
    modality_id,
    clients_per_modality,
    label_skew_alpha,
    min_labels_per_client,
    seed,
):
    rng = np.random.default_rng(seed + modality_id * 100)
    num_classes = int(y.max().item()) + 1
    samples_per_client = len(y) // clients_per_modality

    indices_by_class = {c: np.where(y.numpy() == c)[0].tolist() for c in range(num_classes)}
    for c in indices_by_class:
        rng.shuffle(indices_by_class[c])

    client_indices = [[] for _ in range(clients_per_modality)]
    for c in range(num_classes):
        pool = indices_by_class[c]
        if not pool:
            continue
        w = _dirichlet_weights(clients_per_modality, label_skew_alpha, rng)
        counts = np.floor(w * len(pool)).astype(int)
        while counts.sum() < len(pool):
            counts[rng.integers(0, clients_per_modality)] += 1
        start = 0
        for cid in range(clients_per_modality):
            end = start + counts[cid]
            client_indices[cid].extend(pool[start:end])
            start = end

    for cid in range(clients_per_modality):
        cur = client_indices[cid]
        labels = set(int(y[i]) for i in cur)
        need = max(0, min_labels_per_client - len(labels))
        if need > 0:
            missing = [c for c in range(num_classes) if c not in labels]
            rng.shuffle(missing)
            for c in missing[:need]:
                candidates = np.where(y.numpy() == c)[0]
                if len(candidates) > 0:
                    cur.append(int(candidates[rng.integers(0, len(candidates))]))
        client_indices[cid] = cur

    fixed = []
    for cid in range(clients_per_modality):
        arr = client_indices[cid]
        if len(arr) >= samples_per_client:
            arr = rng.choice(arr, size=samples_per_client, replace=False).tolist()
        else:
            arr = rng.choice(arr, size=samples_per_client, replace=True).tolist()
        fixed.append(arr)

    clients = []
    for local_cid, idxs in enumerate(fixed):
        idx = torch.tensor(idxs, dtype=torch.long)
        clients.append(
            {
                "client_id": f"m{modality_id}_c{local_cid}",
                "modality_id": modality_id,
                "x": x_mod[idx],
                "y": y[idx],
                "gt_cluster": modality_id,
                "input_dim": int(x_mod.shape[1]),
            }
        )
    return clients


def build_client_pool(train_set, cfg):
    all_clients = []
    min_labels = max(2, int(cfg["num_classes"] * cfg["min_labels_per_client_ratio"]))
    for m in range(cfg["num_modalities"]):
        clients = partition_modality_to_clients(
            x_mod=train_set["modalities"][m],
            y=train_set["labels"],
            modality_id=m,
            clients_per_modality=cfg["clients_per_modality"],
            label_skew_alpha=cfg["label_skew_alpha"],
            min_labels_per_client=min_labels,
            seed=cfg["seed"],
        )
        all_clients.extend(clients)
    return all_clients
