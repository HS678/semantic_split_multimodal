import configparser
import json
import re
import shutil
from pathlib import Path


_INTEGER = re.compile(r"^[+-]?\d+$")
_FLOAT = re.compile(
    r"^[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?$"
)


def _parse_value(raw_value: str):
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if value.startswith(("[", "{", '"')):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON-style config value: {value!r}") from exc
    if _INTEGER.fullmatch(value):
        return int(value)
    if _FLOAT.fullmatch(value):
        return float(value)
    return value


def _section_target(cfg: dict, section: str) -> dict:
    if section == "config":
        return cfg
    target = cfg
    for part in section.split("."):
        if not part:
            raise ValueError(f"Invalid empty config section component: {section!r}")
        existing = target.get(part)
        if existing is None:
            existing = {}
            target[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Config section {section!r} conflicts with scalar key {part!r}.")
        target = existing
    return target


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict:
    config_path = Path(path)
    if config_path.suffix.lower() != ".config":
        raise ValueError(f"Config file must use the .config extension: {config_path}")
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = " -> ".join(str(item) for item in [*seen, config_path])
        raise ValueError(f"Circular config extends chain: {chain}")
    seen.add(config_path)

    parser = configparser.ConfigParser(
        interpolation=None,
        delimiters=("=",),
        comment_prefixes=("#", ";"),
        inline_comment_prefixes=None,
        strict=True,
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    with config_path.open("r", encoding="utf-8-sig") as handle:
        parser.read_file(handle)
    if "config" not in parser:
        raise ValueError(f"Config file must contain a [config] section: {config_path}")

    cfg = {}
    for section in parser.sections():
        target = _section_target(cfg, section)
        for key, raw_value in parser.items(section, raw=True):
            clean_key = key.strip()
            if not clean_key:
                raise ValueError(f"Config section {section!r} contains an empty key.")
            target[clean_key] = _parse_value(raw_value)
    extends = cfg.pop("extends", None)
    if extends is None:
        return cfg
    parent_path = Path(str(extends))
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = load_config(parent_path, _seen=seen)
    return _deep_merge(parent, cfg)


def _class_weighting_mode(value) -> str:
    """新格式 class_weighting 用 true/false：true=inverse_sqrt，false=none。"""
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
    raise ValueError(f"class_weighting must be true/false, got {value!r}.")


def split_protocol_for_fold(dataset_type: str, fold: int) -> str:
    from MSL.data.dataset_defaults import DATASET_DEFAULTS

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


def normalize_experiment_config(cfg: dict) -> dict:
    """把 6 段实验配置（config/partition/cluster/train/d2d/other）归一化为内部结构。

    内部结构沿用既有代码读取的字段：dataset/partition/pretrain/fingerprint/
    cluster/cluster.adaptive/training/model.encoder/model.encoders/binding/
    fusion/fusion.mmbind/evaluation/d2d/fingerprint_visualization。
    [config] 段由解析器提升到顶层（experiment_name/seed/device/num_classes/base_dir）。
    每个数据集的固定参数从 DATASET_DEFAULTS 内置，config 只覆盖要切换/要调的字段。
    """
    cfg = dict(cfg)
    if "training" in cfg and "dataset" in cfg:
        # 已是内部结构（或旧格式合并结果），直接返回。
        return cfg
    from MSL.data.dataset_defaults import DATASET_DEFAULTS, DEFAULT_ADAPTIVE
    partition = dict(cfg.pop("partition", {}))
    cluster = dict(cfg.pop("cluster", {}))
    train = dict(cfg.pop("train", {}))
    d2d = dict(cfg.pop("d2d", {}))
    other = dict(cfg.pop("other", {}))
    dataset_type = partition.get("type")
    if not dataset_type:
        raise ValueError("config [partition].type is required.")
    if dataset_type not in DATASET_DEFAULTS:
        raise ValueError(f"Unknown dataset type: {dataset_type!r}")
    defaults = DATASET_DEFAULTS[dataset_type]

    # ---- dataset / partition ----
    dataset = dict(defaults["dataset"])
    split_protocol = partition.get("split_protocol", dataset.get("split_protocol"))
    dataset.update({"type": dataset_type, "normalize": True})  # 内置：只用 train 统计量标准化
    if split_protocol is not None:
        dataset["split_protocol"] = split_protocol
    for key in ("name", "variant", "processed_root", "feature_recipe", "task", "label_protocol"):
        if key in partition:
            dataset[key] = partition[key]
    cfg["dataset"] = dataset
    cfg["partition"] = {"clients_per_modality": int(partition.get("clients_per_modality", 10))}

    # ---- pretrain / fingerprint / cluster ----
    pretrain = dict(defaults["pretrain"])
    for key in ("pretrain_objective", "pretrain_epochs", "pretrain_batch_size", "pretrain_lr",
                "pretrain_class_weighting", "pretrain_weight_decay", "pretrain_max_grad_norm"):
        if key in cluster:
            field = key[len("pretrain_"):]
            pretrain[field] = (
                _class_weighting_mode(cluster[key])
                if field == "class_weighting"
                else cluster[key]
            )
    cfg["pretrain"] = pretrain
    cfg["fingerprint"] = {
        "type": cluster.get("fingerprint_type", defaults["fingerprint_type"]),
        "batch_size": int(cluster.get("fingerprint_batch_size", 64)),
        "max_batches": cluster.get("fingerprint_max_batches", 4),
    }
    adaptive = dict(DEFAULT_ADAPTIVE)
    adaptive.update(dict(defaults.get("cluster_adaptive", {})))
    adaptive.update(
        {
            key[len("adaptive_"):]: value
            for key, value in cluster.items()
            if key.startswith("adaptive_")
        }
    )
    cfg["cluster"] = {
        "method": cluster.get("method", "adaptive_isodata"),
        "known_k": cluster.get("known_k"),
        "adaptive": adaptive,
    }

    # ---- training / encoder / binding / fusion / evaluation ----
    cfg["training"] = {
        "cluster_assignment_source": train.get("cluster_assignment_source", "pred_cluster"),
        "scheduler": train.get("scheduler", "balanced_cluster_round_robin"),
        "global_rounds": int(train.get("global_rounds", defaults.get("default_global_rounds", 200))),
        "local_steps": int(train.get("local_steps", 1)),
        "clients_per_cluster_per_round": int(train.get("clients_per_cluster_per_round", 4)),
        "batch_size": int(train.get("batch_size", defaults["training"]["batch_size"])),
        "eval_batch_size": int(train.get("eval_batch_size", defaults["training"]["eval_batch_size"])),
        "client_lr": float(train.get("client_lr", defaults["training"]["client_lr"])),
        "server_lr": float(train.get("server_lr", defaults["training"]["server_lr"])),
        "client_weight_decay": float(train.get("client_weight_decay", 0.0001)),
        "server_weight_decay": float(train.get("server_weight_decay", 0.0001)),
        "max_grad_norm": train.get("max_grad_norm", 5.0),
        "class_weighting": _class_weighting_mode(
            train.get("class_weighting", defaults["training"]["class_weighting"])
        ),
    }
    encoders = dict(train.pop("encoders", defaults["encoders"]))
    encoder = dict(defaults["encoder"])
    encoder.update(
        {
            key[len("encoder_"):]: value
            for key, value in train.items()
            if key.startswith("encoder_")
        }
    )
    cfg["model"] = {"encoder": encoder, "encoders": encoders}
    cfg["binding"] = {
        "type": train.get("binding_type", "label_random"),
        "batch_size": int(train.get("binding_batch_size", defaults["binding_batch_size"])),
    }
    cfg["fusion"] = {
        "type": train.get("fusion_type", "concat_mlp"),
        "training_objective": train.get("fusion_training_objective", "label_random_ce"),
        "adapter_dim": int(train.get("fusion_adapter_dim", 128)),
        "hidden_dim": int(train.get("fusion_hidden_dim", 256)),
        "num_layers": int(train.get("fusion_num_layers", 2)),
        "dropout": float(train.get("fusion_dropout", defaults["fusion_dropout"])),
        "mmbind": {
            "temperature": float(train.get("mmbind_temperature", defaults["mmbind"]["temperature"])),
            "contrastive_weight": float(train.get("mmbind_contrastive_weight", defaults["mmbind"]["contrastive_weight"])),
            "heterogeneous_ce_weight": float(train.get("mmbind_heterogeneous_ce_weight", defaults["mmbind"]["heterogeneous_ce_weight"])),
        },
    }
    cfg["evaluation"] = {"run_test": bool(train.get("run_test", True))}

    # ---- d2d ----
    cfg["d2d"] = {"enabled": bool(d2d.get("enabled", False))}

    # ---- other -> fingerprint_visualization ----
    visualization = {
        "enabled": bool(other.get("fingerprint_visualization", True)),
        "method": other.get("method", "pca"),
        "standardize": bool(other.get("standardize", True)),
        "show_client_ids": bool(other.get("show_client_ids", False)),
        "show_ellipses": bool(other.get("show_ellipses", True)),
        "ellipse_confidence": float(other.get("ellipse_confidence", 0.95)),
        "png_dpi": int(other.get("png_dpi", 600)),
    }
    cfg["fingerprint_visualization"] = visualization

    cfg.setdefault("num_classes", defaults["num_classes"])
    cfg.setdefault("encoder_hidden_dim", 128)
    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "auto")
    return cfg


def _format_value(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    text = str(value)
    if "\n" in text or "\r" in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def _flatten_sections(mapping: dict, prefix: str = ""):
    scalars = {}
    nested = []
    for key, value in mapping.items():
        if isinstance(value, dict):
            section = f"{prefix}.{key}" if prefix else str(key)
            nested.extend(_flatten_sections(value, section))
        else:
            scalars[str(key)] = value
    if prefix:
        return [(prefix, scalars), *nested]
    return [("config", scalars), *nested]


def write_config(cfg: dict, path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.suffix.lower() != ".config":
        raise ValueError(f"Config snapshot must use the .config extension: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, values in _flatten_sections(cfg):
        if not values and section != "config":
            continue
        if lines:
            lines.append("")
        lines.append(f"[{section}]")
        lines.extend(f"{key}={_format_value(value)}" for key, value in values.items())
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def save_config_artifacts(source_path: str | Path, resolved_cfg: dict, output_dir: str | Path) -> dict:
    source = Path(source_path).resolve()
    if source.suffix.lower() != ".config" or not source.is_file():
        raise ValueError(f"Source config must be an existing .config file: {source}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_copy = destination / "source_config.config"
    resolved_snapshot = destination / "resolved_config.config"
    shutil.copy2(source, source_copy)
    write_config(resolved_cfg, resolved_snapshot)
    return {
        "source_config": str(source_copy),
        "resolved_config": str(resolved_snapshot),
    }
