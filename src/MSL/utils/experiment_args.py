import argparse
import json
from pathlib import Path

from MSL.data.dataset_defaults import DATASET_DEFAULTS, DEFAULT_ADAPTIVE


DATASET_CHOICES = tuple(DATASET_DEFAULTS.keys())


def add_experiment_args(parser: argparse.ArgumentParser, *, baseline: bool = False, include_seed: bool = True):
    """Add the single official experiment CLI.

    The project no longer reads external .config files. Dataset defaults live in
    DATASET_DEFAULTS, and every runtime change is made with an explicit CLI flag.
    """
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True, help="Dataset name.")
    parser.add_argument("--fold", type=int, help="Fold index for datasets with a fold template.")
    parser.add_argument("--split-protocol", help="Override dataset.split_protocol directly.")

    parser.add_argument("--clients", type=int, default=10, help="Clients per discovered modality.")
    parser.add_argument("--global-rounds", type=int, help="Training global rounds.")
    parser.add_argument("--local-steps", type=int, default=1, help="Local client steps per round.")
    parser.add_argument("--client-lr", type=float, help="Client encoder learning rate.")
    parser.add_argument("--server-lr", type=float, help="Server fusion/classifier learning rate.")
    parser.add_argument("--batch-size", type=int, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, help="Evaluation batch size.")
    parser.add_argument("--clients-per-cluster-per-round", type=int, default=2, help="Scheduler client budget.")

    parser.add_argument("--pretrain-epochs", type=int, help="Stage2 client pretraining epochs.")
    parser.add_argument("--pretrain-lr", type=float, help="Stage2 pretraining learning rate.")
    parser.add_argument("--fingerprint-type", choices=["signal", "logit", "hybrid"], help="Stage2 fingerprint type.")

    parser.add_argument(
        "--cluster-assignment-source",
        choices=["pred_cluster", "true_cluster"],
        default="pred_cluster",
        help="Cluster source for Stage3. Use true_cluster only for oracle sanity checks.",
    )
    parser.add_argument(
        "--scheduler",
        default="random" if baseline else "balanced_cluster_round_robin",
        help="Stage3 client scheduler.",
    )
    parser.add_argument(
        "--fusion-training-objective",
        choices=["label_random_ce", "mmbind_weighted_contrastive"],
        default="mmbind_weighted_contrastive",
        help="Fusion training objective.",
    )
    parser.add_argument("--binding-type", default="label_random", help="Semantic pseudo binding type.")

    if include_seed:
        parser.add_argument("--seed", type=int, default=42, help="Experiment seed.")
    parser.add_argument("--device", default="auto", help="Device selector, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit.")


def load_experiment_config_from_args(args, *, baseline: bool = False) -> dict:
    cfg = build_experiment_config(
        dataset_type=args.dataset,
        baseline=baseline,
        seed=getattr(args, "seed", 42),
        device=args.device,
        clients=args.clients,
        scheduler=args.scheduler,
        cluster_assignment_source=args.cluster_assignment_source,
        fusion_training_objective=args.fusion_training_objective,
        binding_type=args.binding_type,
        local_steps=args.local_steps,
        clients_per_cluster_per_round=args.clients_per_cluster_per_round,
    )
    cfg = apply_experiment_overrides(cfg, fold=args.fold, split_protocol=args.split_protocol)
    cfg = _apply_cli_overrides(cfg, args)
    return cfg


def build_experiment_config(
    *,
    dataset_type: str,
    baseline: bool = False,
    seed: int = 42,
    device: str = "auto",
    clients: int = 10,
    scheduler: str | None = None,
    cluster_assignment_source: str = "pred_cluster",
    fusion_training_objective: str = "mmbind_weighted_contrastive",
    binding_type: str = "label_random",
    local_steps: int = 1,
    clients_per_cluster_per_round: int = 2,
) -> dict:
    key = str(dataset_type).strip().lower()
    if key not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")
    defaults = DATASET_DEFAULTS[key]
    scheduler = scheduler or ("random" if baseline else "balanced_cluster_round_robin")
    base_dir = "./results/baseline/randomSL" if baseline else "./results/MSL"
    experiment_suffix = "random_sl" if baseline else "msl"

    dataset = dict(defaults["dataset"])
    dataset.update({"type": key, "normalize": True})

    pretrain = dict(defaults["pretrain"])
    adaptive = dict(DEFAULT_ADAPTIVE)
    adaptive.update(dict(defaults.get("cluster_adaptive", {})))

    training_defaults = defaults["training"]
    cfg = {
        "experiment_name": f"{key}_{experiment_suffix}",
        "base_dir": base_dir,
        "seed": int(seed),
        "device": str(device),
        "num_classes": int(defaults["num_classes"]),
        "encoder_hidden_dim": 128,
        "dataset": dataset,
        "partition": {"clients_per_modality": int(clients)},
        "pretrain": pretrain,
        "fingerprint": {
            "type": defaults["fingerprint_type"],
            "batch_size": 64,
            "max_batches": 4,
        },
        "cluster": {
            "method": "adaptive_isodata",
            "known_k": None,
            "adaptive": adaptive,
        },
        "training": {
            "cluster_assignment_source": str(cluster_assignment_source),
            "scheduler": str(scheduler),
            "global_rounds": int(defaults.get("default_global_rounds", 200)),
            "local_steps": int(local_steps),
            "clients_per_cluster_per_round": int(clients_per_cluster_per_round),
            "batch_size": int(training_defaults["batch_size"]),
            "eval_batch_size": int(training_defaults["eval_batch_size"]),
            "client_lr": float(training_defaults["client_lr"]),
            "server_lr": float(training_defaults["server_lr"]),
            "client_weight_decay": 0.0001,
            "server_weight_decay": 0.0001,
            "max_grad_norm": 5.0,
            "class_weighting": _class_weighting_mode(training_defaults["class_weighting"]),
        },
        "model": {
            "encoder": dict(defaults["encoder"]),
            "encoders": dict(defaults["encoders"]),
        },
        "binding": {
            "type": str(binding_type),
            "batch_size": int(defaults["binding_batch_size"]),
        },
        "fusion": {
            "type": "concat_mlp",
            "training_objective": str(fusion_training_objective),
            "adapter_dim": 128,
            "hidden_dim": 256,
            "num_layers": 2,
            "dropout": float(defaults["fusion_dropout"]),
            "mmbind": {
                "temperature": float(defaults["mmbind"]["temperature"]),
                "contrastive_weight": float(defaults["mmbind"]["contrastive_weight"]),
                "heterogeneous_ce_weight": float(defaults["mmbind"]["heterogeneous_ce_weight"]),
            },
        },
        "evaluation": {"run_test": True},
        "d2d": {"enabled": False},
        "fingerprint_visualization": {
            "enabled": True,
            "method": "pca",
            "standardize": True,
            "show_client_ids": False,
            "show_ellipses": True,
            "ellipse_confidence": 0.95,
            "png_dpi": 600,
        },
    }
    return cfg


def split_protocol_for_fold(dataset_type: str, fold: int) -> str:
    key = str(dataset_type).strip().lower()
    if key not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")
    defaults = DATASET_DEFAULTS[key]
    fold_count = defaults.get("fold_count")
    if fold_count is None:
        raise ValueError(f"Dataset {key!r} does not define folds.")
    fold = int(fold)
    if fold < 1 or fold > int(fold_count):
        raise ValueError(f"Dataset {key!r} fold must be in [1, {fold_count}], got {fold}.")
    template = defaults.get("dataset", {}).get("split_protocol_template")
    if not template:
        raise ValueError(f"Dataset {key!r} does not define split_protocol_template.")
    return str(template).format(fold=fold)


def apply_experiment_overrides(
    cfg: dict,
    *,
    fold: int | None = None,
    split_protocol: str | None = None,
) -> dict:
    if fold is not None and split_protocol is not None:
        raise ValueError("--fold and --split-protocol cannot be used together.")
    if fold is None and split_protocol is None:
        return cfg
    cfg = dict(cfg)
    dataset = dict(cfg.get("dataset", {}))
    dataset_type = dataset.get("type")
    if not dataset_type:
        raise ValueError("dataset.type is required before applying split overrides.")
    if fold is not None:
        split_protocol = split_protocol_for_fold(str(dataset_type), int(fold))
    dataset["split_protocol"] = str(split_protocol)
    cfg["dataset"] = dataset
    cfg["runtime_overrides"] = {
        **dict(cfg.get("runtime_overrides", {})),
        "fold": None if fold is None else int(fold),
        "split_protocol": str(split_protocol),
    }
    return cfg


def save_resolved_config_artifact(resolved_cfg: dict, output_dir: str | Path) -> dict:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "resolved_config.json"
    with snapshot.open("w", encoding="utf-8") as handle:
        json.dump(resolved_cfg, handle, indent=2, ensure_ascii=False, sort_keys=True)
    return {"resolved_config": str(snapshot)}


def stage1_config_snapshot(cfg: dict) -> dict:
    """Keep Stage1 artifacts focused on partition construction inputs."""
    return _copy_selected_sections(
        cfg,
        [
            "experiment_name",
            "base_dir",
            "seed",
            "num_classes",
            "dataset",
            "partition",
            "runtime_overrides",
            "results",
        ],
        config_scope="stage1_partition",
    )


def stage2_config_snapshot(cfg: dict) -> dict:
    """Keep Stage2 artifacts focused on partition and discovery inputs."""
    return _copy_selected_sections(
        cfg,
        [
            "experiment_name",
            "base_dir",
            "seed",
            "device",
            "num_classes",
            "dataset",
            "partition",
            "pretrain",
            "fingerprint",
            "cluster",
            "fingerprint_visualization",
            "runtime_overrides",
            "result",
            "stage2",
        ],
        config_scope="stage2_discovery",
    )


def print_resolved_config(cfg: dict):
    print(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True))


def _apply_cli_overrides(cfg: dict, args) -> dict:
    cfg = dict(cfg)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = int(args.seed)
    if getattr(args, "device", None) is not None:
        cfg["device"] = str(args.device)
    if args.clients is not None:
        cfg["partition"] = {**cfg.get("partition", {}), "clients_per_modality": int(args.clients)}

    training = dict(cfg.get("training", {}))
    _set_if_not_none(training, "global_rounds", args.global_rounds, int)
    _set_if_not_none(training, "local_steps", args.local_steps, int)
    _set_if_not_none(training, "client_lr", args.client_lr, float)
    _set_if_not_none(training, "server_lr", args.server_lr, float)
    _set_if_not_none(training, "batch_size", args.batch_size, int)
    _set_if_not_none(training, "eval_batch_size", args.eval_batch_size, int)
    _set_if_not_none(training, "clients_per_cluster_per_round", args.clients_per_cluster_per_round, int)
    _set_if_not_none(training, "cluster_assignment_source", args.cluster_assignment_source, str)
    _set_if_not_none(training, "scheduler", args.scheduler, str)
    cfg["training"] = training

    pretrain = dict(cfg.get("pretrain", {}))
    _set_if_not_none(pretrain, "epochs", args.pretrain_epochs, int)
    _set_if_not_none(pretrain, "lr", args.pretrain_lr, float)
    cfg["pretrain"] = pretrain

    if args.fingerprint_type is not None:
        cfg["fingerprint"] = {**cfg.get("fingerprint", {}), "type": str(args.fingerprint_type)}
    if args.fusion_training_objective is not None:
        cfg["fusion"] = {
            **cfg.get("fusion", {}),
            "training_objective": str(args.fusion_training_objective),
        }
    if args.binding_type is not None:
        cfg["binding"] = {**cfg.get("binding", {}), "type": str(args.binding_type)}
    return cfg


def _class_weighting_mode(value) -> str:
    if value is None or value is False:
        return "none"
    if value is True:
        return "inverse_sqrt"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "inverse_sqrt"}:
            return "inverse_sqrt"
        if lowered in {"false", "none", "null", ""}:
            return "none"
    raise ValueError(f"class_weighting must be true/false/inverse_sqrt/none, got {value!r}.")


def _set_if_not_none(target: dict, key: str, value, caster):
    if value is not None:
        target[key] = caster(value)


def _copy_selected_sections(cfg: dict, keys: list[str], *, config_scope: str) -> dict:
    snapshot = {"config_scope": config_scope}
    for key in keys:
        if key in cfg:
            snapshot[key] = cfg[key]
    return snapshot
