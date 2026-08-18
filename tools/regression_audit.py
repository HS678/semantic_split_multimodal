import argparse
import csv
import json
from pathlib import Path


PATH_KEYS = {
    "base_dir",
    "checkpoint_dir",
    "cluster_assignment_path",
    "cluster_metadata_path",
    "curve_path",
    "output_dir",
    "result",
    "result_model",
    "run_dir",
    "topology_dir",
}

RUNTIME_KEYS = {
    "d2d",
    "device",
    "git_commit",
    "protocol_hash",
    "config_hash",
    "runtime_overrides",
    "timestamp",
}

TRAIN_LOG_RUNTIME_KEYS = {
    "latency",
}


def load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_train_log(path: Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def scrub_protocol_payload(value):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in PATH_KEYS or key in RUNTIME_KEYS:
                continue
            if key != "training_method" and key.endswith("_method"):
                key = "training_method"
            out[key] = scrub_protocol_payload(item)
        return out
    if isinstance(value, list):
        return [scrub_protocol_payload(item) for item in value]
    return value


def scrub_runtime_payload(value):
    if isinstance(value, dict):
        return {
            key: scrub_runtime_payload(item)
            for key, item in value.items()
            if key not in PATH_KEYS and key not in RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [scrub_runtime_payload(item) for item in value]
    return value


def compare_bytes(old: Path, new: Path) -> bool:
    return Path(old).read_bytes() == Path(new).read_bytes()


def compare_optional_bytes(old: Path, new: Path) -> bool | None:
    old_exists = Path(old).exists()
    new_exists = Path(new).exists()
    if not old_exists and not new_exists:
        return None
    if old_exists != new_exists:
        return False
    return compare_bytes(old, new)


def find_result_file(run_dir: Path) -> Path:
    current = Path(run_dir) / "result.json"
    if current.exists():
        return current
    legacy = sorted(Path(run_dir).glob("*_result.json"))
    if len(legacy) == 1:
        return legacy[0]
    raise FileNotFoundError(f"Expected result.json or one *_result.json under {run_dir}.")


def first_row_diff(old_rows: list[dict], new_rows: list[dict], keys: list[str] | None = None, ignore_keys=None):
    ignore_keys = set(ignore_keys or [])
    if len(old_rows) != len(new_rows):
        return {"row_count": [len(old_rows), len(new_rows)]}
    for index, (old, new) in enumerate(zip(old_rows, new_rows), start=1):
        compare_keys = keys or sorted(set(old) | set(new))
        for key in compare_keys:
            if key in ignore_keys:
                continue
            if old.get(key) != new.get(key):
                return {"row": index, "column": key, "old": old.get(key), "new": new.get(key)}
    return None


def audit(old_dir: Path, new_dir: Path) -> dict:
    old_dir = Path(old_dir)
    new_dir = Path(new_dir)
    old_result_path = find_result_file(old_dir)
    new_result_path = find_result_file(new_dir)
    old_result = load_json(old_result_path)
    new_result = load_json(new_result_path)
    old_metrics = load_json(old_dir / "final_metrics.json")
    new_metrics = load_json(new_dir / "final_metrics.json")
    old_log = read_train_log(old_dir / "train_log.csv")
    new_log = read_train_log(new_dir / "train_log.csv")
    protocol_diff = scrub_protocol_payload(old_result.get("config_snapshot", {})) != scrub_protocol_payload(
        new_result.get("config_snapshot", {})
    )
    selected_diff = first_row_diff(old_log, new_log, ["selected_client_ids"])
    train_log_diff = first_row_diff(old_log, new_log, ignore_keys=TRAIN_LOG_RUNTIME_KEYS)
    metric_diff = scrub_runtime_payload(old_metrics) != scrub_runtime_payload(new_metrics)
    assignment_results = {}
    for rel in [
        "topology/pred_cluster.csv",
        "topology/raw_cluster_assignment.csv",
        "topology/true_cluster.csv",
    ]:
        result = compare_optional_bytes(old_dir / rel, new_dir / rel)
        if result is not None:
            assignment_results[rel] = result
    return {
        "old_dir": str(old_dir),
        "new_dir": str(new_dir),
        "old_device": old_metrics.get("device"),
        "new_device": new_metrics.get("device"),
        "cluster_assignment_equal": all(assignment_results.values()) if assignment_results else True,
        "cluster_assignment_files": assignment_results,
        "selected_clients_equal": selected_diff is None,
        "selected_clients_first_diff": selected_diff,
        "train_log_equal": train_log_diff is None,
        "train_log_first_diff": train_log_diff,
        "final_metrics_equal": not metric_diff,
        "old_final_metrics": {
            "test_accuracy": old_metrics.get("test_accuracy"),
            "test_macro_f1": old_metrics.get("test_macro_f1"),
            "test_loss": old_metrics.get("test_loss"),
        },
        "new_final_metrics": {
            "test_accuracy": new_metrics.get("test_accuracy"),
            "test_macro_f1": new_metrics.get("test_macro_f1"),
            "test_loss": new_metrics.get("test_loss"),
        },
        "protocol_params_equal": not protocol_diff,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Compare one fixed pre/post-refactor training run.")
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--new-dir", required=True)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result = audit(Path(args.old_dir), Path(args.new_dir))
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    if not all(
        [
            result["cluster_assignment_equal"],
            result["selected_clients_equal"],
            result["train_log_equal"],
            result["final_metrics_equal"],
            result["protocol_params_equal"],
        ]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
