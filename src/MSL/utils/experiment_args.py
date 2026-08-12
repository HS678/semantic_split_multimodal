import argparse
import json
from pathlib import Path

from MSL.data.dataset_defaults import DATASET_DEFAULTS
from MSL.utils.config import apply_experiment_overrides, load_config, normalize_experiment_config


DATASET_CHOICES = tuple(DATASET_DEFAULTS.keys())


def add_experiment_args(parser: argparse.ArgumentParser, *, baseline: bool = False, include_seed: bool = True):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Path to INI-style .config file.")
    source.add_argument("--dataset", choices=DATASET_CHOICES, help="Dataset name; loads built-in defaults.")

    parser.add_argument("--fold", type=int, help="Override dataset.split_protocol from the dataset fold template.")
    parser.add_argument("--split-protocol", help="Override dataset.split_protocol directly.")
    parser.add_argument("--clients", type=int, help="Override partition.clients_per_modality.")
    parser.add_argument("--global-rounds", type=int, help="Override training.global_rounds.")
    parser.add_argument("--client-lr", type=float, help="Override training.client_lr.")
    parser.add_argument("--server-lr", type=float, help="Override training.server_lr.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--eval-batch-size", type=int, help="Override training.eval_batch_size.")
    parser.add_argument("--clients-per-cluster-per-round", type=int, help="Override scheduler client budget.")
    parser.add_argument("--pretrain-epochs", type=int, help="Override pretrain.epochs.")
    parser.add_argument("--pretrain-lr", type=float, help="Override pretrain.lr.")
    parser.add_argument("--fingerprint-type", help="Override fingerprint.type.")
    parser.add_argument(
        "--fusion-training-objective",
        choices=["label_random_ce", "mmbind_weighted_contrastive"],
        help="Override fusion.training_objective.",
    )
    parser.add_argument(
        "--cluster-assignment-source",
        choices=["pred_cluster", "true_cluster"],
        help="Override training.cluster_assignment_source.",
    )
    parser.add_argument("--scheduler", default="random" if baseline else None, help="Override training.scheduler.")
    if include_seed:
        parser.add_argument("--seed", type=int, help="Override experiment seed.")
    parser.add_argument("--print-config", action="store_true", help="Print the fully resolved config and exit.")


def load_experiment_config_from_args(args, *, baseline: bool = False):
    source_path = None
    if args.config:
        source_path = Path(args.config)
        cfg = normalize_experiment_config(load_config(source_path))
    else:
        cfg = normalize_experiment_config(_dataset_config(args.dataset, baseline=baseline))

    cfg = apply_experiment_overrides(cfg, fold=args.fold, split_protocol=args.split_protocol)
    cfg = _apply_cli_overrides(cfg, args)
    return cfg, source_path


def print_resolved_config(cfg: dict):
    print(json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True))


def _dataset_config(dataset_type: str, *, baseline: bool) -> dict:
    defaults = DATASET_DEFAULTS[dataset_type]
    scheduler = "random" if baseline else "balanced_cluster_round_robin"
    base_dir = "./results/baseline/randomSL" if baseline else "./results/MSL"
    experiment_suffix = "random_sl" if baseline else "msl"
    partition = {
        "type": dataset_type,
        "clients_per_modality": 10,
    }
    split_protocol = defaults.get("dataset", {}).get("split_protocol")
    if split_protocol:
        partition["split_protocol"] = split_protocol
    return {
        "experiment_name": f"{dataset_type}_{experiment_suffix}",
        "base_dir": base_dir,
        "partition": partition,
        "train": {
            "cluster_assignment_source": "pred_cluster",
            "scheduler": scheduler,
            "clients_per_cluster_per_round": 2,
            "fusion_training_objective": "mmbind_weighted_contrastive",
            "global_rounds": int(defaults.get("default_global_rounds", 200)),
        },
        "d2d": {"enabled": False},
        "other": {"fingerprint_visualization": True},
    }


def _apply_cli_overrides(cfg: dict, args) -> dict:
    cfg = dict(cfg)
    if getattr(args, "seed", None) is not None:
        cfg["seed"] = int(args.seed)
    if args.clients is not None:
        cfg["partition"] = {**cfg.get("partition", {}), "clients_per_modality": int(args.clients)}

    training = dict(cfg.get("training", {}))
    _set_if_not_none(training, "global_rounds", args.global_rounds, int)
    _set_if_not_none(training, "client_lr", args.client_lr, float)
    _set_if_not_none(training, "server_lr", args.server_lr, float)
    _set_if_not_none(training, "batch_size", args.batch_size, int)
    _set_if_not_none(training, "eval_batch_size", args.eval_batch_size, int)
    _set_if_not_none(training, "clients_per_cluster_per_round", args.clients_per_cluster_per_round, int)
    _set_if_not_none(training, "cluster_assignment_source", args.cluster_assignment_source, str)
    if args.scheduler is not None:
        training["scheduler"] = str(args.scheduler)
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
    return cfg


def _set_if_not_none(target: dict, key: str, value, caster):
    if value is not None:
        target[key] = caster(value)
