import csv
import math
from pathlib import Path

import pytest

from experiments.convergence import aggregate_common_targets
from experiments.msl.run_all import aggregate as aggregate_training


def _write_curve(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "test_accuracy", "test_macro_f1", "test_weighted_f1"])
        writer.writeheader()
        for idx, value in enumerate(values):
            writer.writerow(
                {
                    "round": int((idx + 1) * 10),
                    "test_accuracy": float(value),
                    "test_macro_f1": float(value),
                    "test_weighted_f1": float(value),
                }
            )


def _manifest(tmp_path: Path, fold= None, seed=1):
    root = tmp_path / "runs"
    curves = {
        "oracle": [0.20, 0.40, 0.60, 0.80, 1.00],
        "randomsl": [0.20, 0.45, 0.55, 0.63, 0.75],
        "kmeans2": [0.10, 0.30, 0.50, 0.63, 0.81],
        "kmeans3": [0.10, 0.20, 0.30, 0.40, 0.47],
        "kmeans4": [0.70, 0.70, 0.70, 0.70, 0.70],
        "kmeans5": [0.10, 0.70, 0.70, 0.70, 0.70],
        "auto_kmeans": [0.10, 0.20, 0.80, 0.80, 0.80],
        "gmm_bic": [0.10, 0.20, 0.30, 0.80, 0.80],
        "ours": [0.20, 0.60, 0.70, 0.80, 0.90],
    }
    records = []
    for method, values in curves.items():
        curve_file = root / "uci_har" / str(seed) / method / "test_curve.csv"
        _write_curve(curve_file, values)
        records.append(
            {
                "dataset": "uci_har",
                "fold": fold,
                "seed": seed,
                "method": method,
                "evaluation_mode": "curve",
                "source_kind": "synthetic",
                "run_dir": str(curve_file.parent),
                "curve_file": str(curve_file),
            }
        )
    return records


def test_common_targets_shared_and_from_oracle_last3_mean(tmp_path):
    payload = aggregate_common_targets(_manifest(tmp_path))
    records = payload["records"]

    targets = {(row["target60"], row["target70"], row["target80"]) for row in records}
    assert len(targets) == 1
    oracle_stable = (0.60 + 0.80 + 1.00) / 3
    row = records[0]
    assert row["oracle_stable_macro_f1"] == pytest.approx(oracle_stable)
    assert row["target60"] == pytest.approx(0.60 * oracle_stable)
    assert row["target70"] == pytest.approx(0.70 * oracle_stable)
    assert row["target80"] == pytest.approx(0.80 * oracle_stable)
    assert [row["oracle_last3_round_1"], row["oracle_last3_round_2"], row["oracle_last3_round_3"]] == [30, 40, 50]


def test_common_target_first_crossing_and_not_reached(tmp_path):
    payload = aggregate_common_targets(_manifest(tmp_path))
    rows = {row["method"]: row for row in payload["records"]}

    assert rows["randomsl"]["R60"] == 30.0
    assert rows["randomsl"]["R70"] == 40.0
    assert rows["randomsl"]["R80"] == 50.0
    assert rows["randomsl"]["macro_f1_at_R70"] == pytest.approx(0.63)
    assert rows["kmeans3"]["reached60"] is False
    assert rows["kmeans3"]["status60"] == "not_reached"
    assert math.isnan(rows["kmeans3"]["R60"])
    assert math.isnan(rows["kmeans3"]["macro_f1_at_R60"])


def test_common_target_speedup_is_matched_first(tmp_path):
    payload = aggregate_common_targets(_manifest(tmp_path))
    rows = {row["method"]: row for row in payload["records"]}

    assert rows["randomsl"]["S80"] == pytest.approx(1.0)
    assert rows["kmeans2"]["R80"] == 50.0
    assert rows["kmeans2"]["S80"] == pytest.approx(1.0)
    assert rows["auto_kmeans"]["R80"] == 30.0
    assert rows["auto_kmeans"]["S80"] == pytest.approx(50.0 / 30.0)
    assert math.isnan(rows["kmeans3"]["S80"])


def test_partial_reach_aggregation_reports_reach_rate(tmp_path):
    records = _manifest(tmp_path, seed=1) + _manifest(tmp_path, seed=2)
    # Make kmeans2 miss all targets in the second matched run.
    miss_file = Path([row for row in records if row["method"] == "kmeans2" and row["seed"] == 2][0]["curve_file"])
    _write_curve(miss_file, [0.1, 0.1, 0.1, 0.1, 0.1])

    summary = aggregate_common_targets(records)["summary"]["uci_har_kmeans2"]

    assert summary["reach_rate80"] == pytest.approx(0.5)
    assert summary["R80_mean"] == pytest.approx(50.0)
    assert summary["R80_std"] == pytest.approx(0.0)


def test_table_iv_fmc_and_eur_aggregation():
    summary = aggregate_training(
        [
            {
                "dataset": "mhealth",
                "method": "ours",
                "status": "success",
                "metrics": {
                    "modality_full_coverage_rate": 0.75,
                    "local_step_binding_success_rate": 0.5,
                },
            },
            {
                "dataset": "mhealth",
                "method": "ours",
                "status": "success",
                "metrics": {
                    "modality_full_coverage_rate": 1.0,
                    "local_step_binding_success_rate": 0.75,
                },
            },
        ]
    )

    item = summary["mhealth_ours"]
    assert item["FMC_mean"] == pytest.approx(0.875)
    assert item["EUR_mean"] == pytest.approx(0.625)
    assert item["FMC_std"] > 0.0
    assert item["EUR_std"] > 0.0
