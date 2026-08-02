import importlib.util
import json
from pathlib import Path

from semantic_split_multimodal.utils.config import write_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_formal_results.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("summarize_formal_results", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, dataset: str, fold: int | None, value: float, valid=True):
    run = root / "experiments" / "oracle_true_cluster" / dataset / f"cfg-{fold or 0}" / "seed-101" / "attempt-01"
    run.mkdir(parents=True)
    dataset_cfg = {"type": dataset, "split_protocol": f"split-{fold or 0}"}
    if fold is not None:
        dataset_cfg["test_sessions"] = [fold]
    write_config(
        {
            "seed": 101,
            "dataset": dataset_cfg,
            "stage3": {"attempt": 1, "config_signature": f"cfg-{fold or 0}"},
        },
        run / "resolved_config.config",
    )
    (run / "stage3_metadata.json").write_text(
        json.dumps({"status": "success" if valid else "failed"}), encoding="utf-8"
    )
    (run / "final_metrics.json").write_text(
        json.dumps(
            {
                "test_eval_status": "success",
                "test_evaluation_count": 1,
                "official_result": {"metrics_file": "final_metrics.json"},
                "test_accuracy": value,
                "test_balanced_accuracy": value - 0.1,
                "test_macro_f1": value - 0.2,
                "test_weighted_f1": value - 0.05,
                "test_binary_f1": None,
                "test_confusion_matrix": [[1, 2], [3, 4]],
                "test_num_eval_samples": 10,
            }
        ),
        encoding="utf-8",
    )


def test_summary_marks_missing_iemocap_folds_and_ignores_failed_runs(tmp_path):
    script = _load_script()
    _write_run(tmp_path, "uci_har", None, 0.8)
    _write_run(tmp_path, "pamap2", None, 0.4, valid=False)
    _write_run(tmp_path, "iemocap", 1, 0.6)
    _write_run(tmp_path, "iemocap", 3, 0.8)

    summary = script.build_summary(tmp_path)
    assert summary["datasets"]["uci_har"]["status"] == "complete"
    assert summary["datasets"]["pamap2"]["status"] == "missing"
    iemocap = summary["datasets"]["iemocap"]
    assert iemocap["status"] == "incomplete"
    assert iemocap["completed_test_sessions"] == [1, 3]
    assert iemocap["missing_test_sessions"] == [2, 4, 5]
    assert iemocap["five_fold_metrics"]["test_accuracy"]["mean"] == 0.7
    assert iemocap["aggregate_confusion_matrix"] == [[2, 4], [6, 8]]


def test_summary_only_marks_iemocap_complete_with_all_five_sessions(tmp_path):
    script = _load_script()
    for fold in range(1, 6):
        _write_run(tmp_path, "iemocap", fold, fold / 10)

    iemocap = script.build_summary(tmp_path)["datasets"]["iemocap"]
    assert iemocap["status"] == "complete"
    assert iemocap["missing_test_sessions"] == []
    assert iemocap["five_fold_metrics"]["test_accuracy"]["values"] == [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
    ]
