# Compact C2 v2 artifact manifest, export, and audit utilities.
import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from experiments.common import DATASET_PROTOCOLS, fold_result_component, formal_run_grid, protocol_hash, seed_result_component, write_json
from experiments.convergence import C2_V2_METHODS, COMMON_TARGET_LEVELS
from experiments.training import expected_training_config_hash, training_run_dir


LEGACY_REUSED_METHODS = {"randomsl", "ours", "oracle"}


def canonical_run_dir(output_root: Path, evaluation_mode: str, dataset: str, fold: int | None, seed: int, method: str) -> Path:
    return Path(output_root) / str(evaluation_mode) / dataset / fold_result_component(fold) / seed_result_component(seed) / method


def build_manifest_records(
    *,
    output_root: Path,
    legacy_curve_root: Path,
    legacy_formal_root: Path,
    new_curve_root: Path,
    new_formal_root: Path,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[dict]:
    datasets = list(DATASET_PROTOCOLS) if datasets is None else list(datasets)
    records = []
    for dataset in datasets:
        for fold, seed in formal_run_grid(dataset, seeds):
            for method in C2_V2_METHODS:
                for mode in ("curve", "formal"):
                    source_root = (
                        legacy_curve_root if mode == "curve" and method in LEGACY_REUSED_METHODS
                        else legacy_formal_root if mode == "formal" and method in LEGACY_REUSED_METHODS
                        else new_curve_root if mode == "curve"
                        else new_formal_root
                    )
                    source_run_dir = training_run_dir(source_root, dataset, fold, int(seed), method, None)
                    expected_dir = canonical_run_dir(output_root, mode, dataset, fold, int(seed), method)
                    records.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "seed": int(seed),
                            "method": method,
                            "evaluation_mode": mode,
                            "source_kind": "legacy_reused" if method in LEGACY_REUSED_METHODS else "v2_new_run",
                            "source_root": str(Path(source_root)),
                            "source_run_dir": str(source_run_dir),
                            "expected_output_path": str(expected_dir),
                            "expected_files": (
                                ["run_meta.json", "train_log.csv", "test_curve.csv"]
                                if mode == "curve"
                                else ["run_meta.json", "final_metrics.json", "formal_test_access.json"]
                            ),
                        }
                    )
    return records


def write_manifest(
    output_root: Path,
    records: list[dict],
    *,
    legacy_curve_root: Path,
    legacy_formal_root: Path,
    new_curve_root: Path,
    new_formal_root: Path,
) -> Path:
    payload = {
        "artifact_contract": "c2_v2_common_oracle_compact",
        "protocol_hash": protocol_hash(),
        "methods": list(C2_V2_METHODS),
        "legacy_reused_methods": sorted(LEGACY_REUSED_METHODS),
        "output_root": str(Path(output_root)),
        "source_roots": {
            "legacy_curve": str(Path(legacy_curve_root)),
            "legacy_formal": str(Path(legacy_formal_root)),
            "new_curve": str(Path(new_curve_root)),
            "new_formal": str(Path(new_formal_root)),
        },
        "runs": records,
    }
    path = Path(output_root) / "manifest.json"
    write_json(path, payload)
    return path


def load_manifest(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("runs"), list):
        raise ValueError("manifest.json must contain a runs list.")
    return payload


def _read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compact_run_meta(record: dict, source_result: dict | None, source_files: dict) -> dict:
    metrics = (source_result or {}).get("metrics") or {}
    return {
        "dataset": record["dataset"],
        "fold": record["fold"],
        "seed": int(record["seed"]),
        "method": record["method"],
        "evaluation_mode": record["evaluation_mode"],
        "source_kind": record["source_kind"],
        "source_run_dir": record["source_run_dir"],
        "source_files": source_files,
        "status": (source_result or {}).get("status"),
        "protocol_hash": (source_result or {}).get("protocol_hash"),
        "config_hash": (source_result or {}).get("config_hash"),
        "git_commit": (source_result or {}).get("git_commit"),
        "clients_per_round": (source_result or {}).get("clients_per_round", metrics.get("clients_per_round")),
        "Rmax": metrics.get("configured_global_rounds"),
        "test_evaluations": metrics.get("test_evaluations"),
        "final_test_evaluation_count": metrics.get("final_test_evaluation_count"),
        "periodic_test_evaluation_count": metrics.get("periodic_test_evaluation_count"),
    }


def export_compact_artifacts(manifest_path: Path) -> list[dict]:
    manifest = load_manifest(manifest_path)
    exported = []
    for record in manifest["runs"]:
        source_dir = Path(record["source_run_dir"])
        target_dir = Path(record["expected_output_path"])
        target_dir.mkdir(parents=True, exist_ok=True)
        result_path = source_dir / "result.json"
        source_result = _read_json(result_path) if result_path.exists() else None
        source_files = {"result": str(result_path) if result_path.exists() else None}

        if record["evaluation_mode"] == "curve":
            for name in ("train_log.csv", "test_curve.csv"):
                source = source_dir / name
                if not source.exists():
                    raise FileNotFoundError(f"Missing source curve artifact: {source}")
                shutil.copy2(source, target_dir / name)
                source_files[name] = str(source)
        else:
            for name in ("final_metrics.json", "formal_test_access.json"):
                source = source_dir / name
                if not source.exists():
                    raise FileNotFoundError(f"Missing source formal artifact: {source}")
                shutil.copy2(source, target_dir / name)
                source_files[name] = str(source)

        write_json(target_dir / "run_meta.json", _compact_run_meta(record, source_result, source_files))
        exported.append({**record, "exported": True})
    return exported


def write_curve_manifest_for_common_targets(manifest_path: Path, output_path: Path) -> Path:
    manifest = load_manifest(manifest_path)
    curve_runs = []
    for record in manifest["runs"]:
        if record["evaluation_mode"] != "curve":
            continue
        run_dir = Path(record["expected_output_path"])
        curve_runs.append(
            {
                "dataset": record["dataset"],
                "fold": record["fold"],
                "seed": int(record["seed"]),
                "method": record["method"],
                "evaluation_mode": "curve",
                "source_kind": record["source_kind"],
                "run_dir": str(run_dir),
                "curve_file": str(run_dir / "test_curve.csv"),
                "v2_output_root": str(Path(manifest["output_root"])),
            }
        )
    write_json(output_path, {"runs": curve_runs})
    return output_path


def _manifest_key(record: dict) -> tuple:
    return (
        record["evaluation_mode"],
        record["dataset"],
        record.get("fold"),
        int(record["seed"]),
        record["method"],
    )


def _discover_canonical_runs(output_root: Path) -> list[dict]:
    runs = []
    for mode in ("curve", "formal"):
        base = Path(output_root) / mode
        if not base.exists():
            continue
        for run_meta in base.glob("*/*/*/*/run_meta.json"):
            parts = run_meta.relative_to(base).parts
            if len(parts) != 5:
                continue
            dataset, fold_name, seed_name, method, _ = parts
            fold = None if fold_name == "fold_00" else int(fold_name.replace("fold_", ""))
            seed = int(seed_name.replace("seed_", ""))
            runs.append(
                {
                    "evaluation_mode": mode,
                    "dataset": dataset,
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "run_dir": str(run_meta.parent),
                }
            )
    return runs


def audit_artifacts(manifest_path: Path, output_root: Path) -> dict:
    manifest = load_manifest(manifest_path)
    expected = manifest["runs"]
    expected_keys = [_manifest_key(record) for record in expected]
    expected_key_set = set(expected_keys)
    duplicate_runs = [key for key in sorted(expected_key_set) if expected_keys.count(key) > 1]
    discovered = _discover_canonical_runs(output_root)
    discovered_keys = {_manifest_key(record) for record in discovered}

    missing_runs = [record for record in expected if _manifest_key(record) not in discovered_keys]
    unexpected_runs = [record for record in discovered if _manifest_key(record) not in expected_key_set or record["method"] not in C2_V2_METHODS]
    missing_files = []
    duplicate_files = []
    provenance_mismatches = []
    curve_formal_contamination = []
    for record in expected:
        run_dir = Path(record["expected_output_path"])
        for name in record["expected_files"]:
            if not (run_dir / name).exists():
                missing_files.append({"run": record, "file": name})
        canonical_counts = {
            "test_curve.csv": len(list(run_dir.glob("**/test_curve.csv"))),
            "final_metrics.json": len(list(run_dir.glob("**/final_metrics.json"))),
            "formal_test_access.json": len(list(run_dir.glob("**/formal_test_access.json"))),
        }
        if record["evaluation_mode"] == "curve":
            if canonical_counts["test_curve.csv"] != 1:
                duplicate_files.append({"run": record, "file": "test_curve.csv", "count": canonical_counts["test_curve.csv"]})
            for forbidden in ("final_metrics.json", "formal_test_access.json"):
                if canonical_counts[forbidden] > 0:
                    curve_formal_contamination.append({"run": record, "file": forbidden})
        else:
            for name in ("final_metrics.json", "formal_test_access.json"):
                if canonical_counts[name] != 1:
                    duplicate_files.append({"run": record, "file": name, "count": canonical_counts[name]})
            if canonical_counts["test_curve.csv"] > 0:
                curve_formal_contamination.append({"run": record, "file": "test_curve.csv"})
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            meta = _read_json(meta_path)
            for field in ("dataset", "fold", "seed", "method", "evaluation_mode"):
                if meta.get(field) != record.get(field):
                    provenance_mismatches.append({"run": record, "field": field, "manifest": record.get(field), "run_meta": meta.get(field)})

    common_fields = [
        "oracle_stable_macro_f1", "target60", "target70", "target80",
        "R60", "R70", "R80",
        "macro_f1_at_R60", "macro_f1_at_R70", "macro_f1_at_R80",
        "reached60", "reached70", "reached80",
        "S60", "S70", "S80",
    ]
    common_missing_fields = []
    records_path = Path(output_root) / "common_targets" / "records.csv"
    if records_path.exists():
        with records_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = sorted(set(common_fields) - set(reader.fieldnames or []))
            if missing:
                common_missing_fields.extend(missing)
    else:
        missing_files.append({"run": "common_targets", "file": "records.csv"})

    problem_lists = [
        missing_runs,
        duplicate_runs,
        unexpected_runs,
        missing_files,
        duplicate_files,
        provenance_mismatches,
        curve_formal_contamination,
        common_missing_fields,
    ]
    verdict = "PASS" if all(not items for items in problem_lists) else "FAIL"
    audit = {
        "expected_runs": len(expected),
        "discovered_runs": len(discovered),
        "missing_runs": missing_runs,
        "duplicate_runs": [str(key) for key in duplicate_runs],
        "unexpected_runs": unexpected_runs,
        "missing_files": missing_files,
        "duplicate_files": duplicate_files,
        "provenance_mismatches": provenance_mismatches,
        "curve_formal_contamination": curve_formal_contamination,
        "missing_common_target_fields": common_missing_fields,
        "verdict": verdict,
    }
    write_json(Path(output_root) / "artifact_audit.json", audit)
    return audit


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build/export/audit compact C2 v2 artifacts.")
    parser.add_argument("--output-root", default="results/c2_v2_common_oracle")
    parser.add_argument("--legacy-curve-root", default="results/c2_curve")
    parser.add_argument("--legacy-formal-root", default="results/c2_formal")
    parser.add_argument("--new-curve-root", default="local/c2_v2_common_oracle_cache/curve")
    parser.add_argument("--new-formal-root", default="local/c2_v2_common_oracle_cache/formal")
    parser.add_argument("--action", choices=("manifest", "export", "curve-manifest", "audit", "all"), required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_root = (ROOT / args.output_root).resolve()
    manifest_path = output_root / "manifest.json"
    if args.action in {"manifest", "all"}:
        records = build_manifest_records(
            output_root=output_root,
            legacy_curve_root=ROOT / args.legacy_curve_root,
            legacy_formal_root=ROOT / args.legacy_formal_root,
            new_curve_root=ROOT / args.new_curve_root,
            new_formal_root=ROOT / args.new_formal_root,
        )
        manifest_path = write_manifest(
            output_root,
            records,
            legacy_curve_root=ROOT / args.legacy_curve_root,
            legacy_formal_root=ROOT / args.legacy_formal_root,
            new_curve_root=ROOT / args.new_curve_root,
            new_formal_root=ROOT / args.new_formal_root,
        )
        print(f"manifest={manifest_path}")
    if args.action in {"export", "all"}:
        exported = export_compact_artifacts(manifest_path)
        print(f"exported={len(exported)}")
    if args.action in {"curve-manifest", "all"}:
        curve_manifest = write_curve_manifest_for_common_targets(manifest_path, output_root / "common_targets" / "run_manifest.json")
        print(f"curve_manifest={curve_manifest}")
    if args.action in {"audit", "all"}:
        audit = audit_artifacts(manifest_path, output_root)
        print(f"artifact_audit={audit['verdict']}")


if __name__ == "__main__":
    main()
