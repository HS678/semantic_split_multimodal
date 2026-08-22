import csv
import json
from pathlib import Path

from experiments.c2_v2_artifacts import (
    audit_artifacts,
    build_manifest_records,
    export_compact_artifacts,
    write_curve_manifest_for_common_targets,
    write_manifest,
)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _source_run(path: Path, mode: str):
    if mode == "curve":
        _write_csv(path / "train_log.csv", ["round", "loss"], [{"round": 1, "loss": 0.0}])
        _write_csv(
            path / "test_curve.csv",
            ["round", "test_accuracy", "test_macro_f1", "test_weighted_f1"],
            [{"round": 10, "test_accuracy": 0.1, "test_macro_f1": 0.1, "test_weighted_f1": 0.1}],
        )
    else:
        _write_json(path / "final_metrics.json", {"test_accuracy": 0.1, "test_macro_f1": 0.1})
        _write_json(path / "formal_test_access.json", {"status": "accessed", "evaluation_mode": "formal"})
    _write_json(
        path / "result.json",
        {
            "status": "success",
            "protocol_hash": "abc",
            "config_hash": "cfg",
            "git_commit": "head",
            "clients_per_round": 4,
            "metrics": {
                "configured_global_rounds": 200,
                "test_evaluations": 1,
                "final_test_evaluation_count": 1 if mode == "formal" else 0,
                "periodic_test_evaluation_count": 0 if mode == "formal" else 20,
            },
        },
    )


def test_v2_manifest_explicitly_enumerates_expected_runs(tmp_path):
    output = tmp_path / "v2"
    records = build_manifest_records(
        output_root=output,
        legacy_curve_root=tmp_path / "legacy_curve",
        legacy_formal_root=tmp_path / "legacy_formal",
        new_curve_root=tmp_path / "new_curve",
        new_formal_root=tmp_path / "new_formal",
        datasets=["uci_har"],
        seeds=[1],
    )

    assert len(records) == 16
    assert {
        "dataset",
        "fold",
        "seed",
        "method",
        "evaluation_mode",
        "expected_output_path",
    } <= set(records[0])
    assert all(str(record["expected_output_path"]).startswith(str(output)) for record in records)


def test_v2_export_compact_artifacts_and_audit_pass(tmp_path):
    output = tmp_path / "v2"
    source = tmp_path / "source"
    records = []
    for mode in ("curve", "formal"):
        run_dir = source / mode / "run"
        _source_run(run_dir, mode)
        records.append(
            {
                "dataset": "uci_har",
                "fold": None,
                "seed": 1,
                "method": "randomsl",
                "evaluation_mode": mode,
                "source_kind": "legacy_reused",
                "source_root": str(source),
                "source_run_dir": str(run_dir),
                "expected_output_path": str(output / mode / "uci_har" / "fold_00" / "seed_1" / "randomsl"),
                "expected_files": ["run_meta.json", "train_log.csv", "test_curve.csv"] if mode == "curve" else ["run_meta.json", "final_metrics.json", "formal_test_access.json"],
            }
        )
    manifest_path = write_manifest(
        output,
        records,
        legacy_curve_root=source,
        legacy_formal_root=source,
        new_curve_root=source,
        new_formal_root=source,
    )

    export_compact_artifacts(manifest_path)
    curve_manifest = write_curve_manifest_for_common_targets(manifest_path, output / "common_targets" / "run_manifest.json")

    assert (output / "curve" / "uci_har" / "fold_00" / "seed_1" / "randomsl" / "test_curve.csv").exists()
    assert (output / "formal" / "uci_har" / "fold_00" / "seed_1" / "randomsl" / "formal_test_access.json").exists()
    assert not (output / "curve" / "uci_har" / "fold_00" / "seed_1" / "randomsl" / "result.json").exists()
    assert curve_manifest.exists()

    _write_csv(
        output / "common_targets" / "records.csv",
        [
            "oracle_stable_macro_f1",
            "target60",
            "target70",
            "target80",
            "R60",
            "R70",
            "R80",
            "macro_f1_at_R60",
            "macro_f1_at_R70",
            "macro_f1_at_R80",
            "reached60",
            "reached70",
            "reached80",
            "S60",
            "S70",
            "S80",
        ],
        [],
    )
    audit = audit_artifacts(manifest_path, output)
    assert audit["verdict"] == "PASS"
