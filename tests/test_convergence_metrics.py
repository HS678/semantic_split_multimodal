import csv
from pathlib import Path

import pytest

from experiments.convergence import aggregate_convergence, compute_curve_metrics
from experiments.msl.run_all import aggregate as aggregate_training
from experiments.training import training_run_dir


def _rows(values):
    return [
        {
            "round": int((idx + 1) * 10),
            "test_accuracy": float(value),
            "test_macro_f1": float(value),
            "test_weighted_f1": float(value),
        }
        for idx, value in enumerate(values)
    ]


def _write_curve(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "test_accuracy", "test_macro_f1", "test_weighted_f1"])
        writer.writeheader()
        writer.writerows(_rows(values))


def test_compute_curve_metrics_last3_and_round_thresholds():
    metrics = compute_curve_metrics(_rows([0.1, 0.3, 0.5, 0.65, 0.75, 0.85, 0.88, 0.89, 0.91, 0.92]))

    assert metrics["M_ref"] == pytest.approx((0.89 + 0.91 + 0.92) / 3)
    assert metrics["R70"] == 40
    assert metrics["R80"] == 50
    assert metrics["R90"] == 60


def test_compute_curve_metrics_fails_with_too_few_checkpoints():
    with pytest.raises(RuntimeError, match="at least 3"):
        compute_curve_metrics(_rows([0.1, 0.2]))


def test_aggregate_convergence_matched_speedup_and_summary(tmp_path):
    root = tmp_path / "results" / "c2_curve"
    random_values = [0.1, 0.3, 0.5, 0.65, 0.75, 0.85, 0.88, 0.89, 0.91, 0.92]
    ours_values = [0.3, 0.75, 0.85, 0.88, 0.89, 0.9, 0.9, 0.91, 0.92, 0.93]
    oracle_values = [0.4, 0.78, 0.86, 0.88, 0.89, 0.9, 0.9, 0.91, 0.92, 0.93]
    for method, values in {
        "randomsl": random_values,
        "ours": ours_values,
        "oracle": oracle_values,
    }.items():
        run_dir = training_run_dir(root, "uci_har", None, 1, method, None)
        _write_curve(run_dir / "test_curve.csv", values)

    payload = aggregate_convergence(root, datasets=["uci_har"], methods=["randomsl", "ours", "oracle"], seeds=[1])

    records = {
        record["method"]: record
        for record in payload["records"]
    }
    assert records["randomsl"]["S90"] == pytest.approx(1.0)
    assert records["ours"]["R90"] == 30
    assert records["ours"]["S90"] == pytest.approx(2.0)
    assert payload["summary"]["uci_har_ours"]["R90_mean"] == pytest.approx(30.0)
    assert payload["summary"]["uci_har_ours"]["S90_mean"] == pytest.approx(2.0)


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
