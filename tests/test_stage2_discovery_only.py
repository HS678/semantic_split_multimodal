import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "stage2_discovery_only.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage2_discovery_only", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage1_dir(tmp_path):
    path = tmp_path / "stage1" / "01_dataset_partition"
    (path / "train_clients").mkdir(parents=True)
    return path


def _cfg():
    return {
        "seed": 42,
        "dataset": {"type": "synthetic_stage2_only"},
        "cluster": {
            "method": "adaptive_isodata",
            "known_k": None,
            "adaptive": {"seeds": [11, 23], "max_iter": 2},
        },
    }


def test_stage2_only_builds_isolated_output_paths_without_touching_stage1(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = tmp_path / "formal_outputs"

    cfg, paths = script.build_stage2_only_run(
        _cfg(),
        stage1_dir=stage1,
        output_root=output_root,
        tag="audit",
        run_type="user_formal",
    )

    assert Path(cfg["partition"]["output_dir"]) == stage1.resolve()
    expected_run = output_root.resolve() / "synthetic_stage2_only" / "audit"
    assert Path(cfg["cluster"]["output_dir"]) == expected_run / "02_cluster_results"
    assert Path(cfg["result"]["output_dir"]) == expected_run / "02_discovery_logs"
    assert not (stage1.parent / "02_cluster_results").exists()
    assert not (stage1.parent / "03_training_evaluation").exists()
    assert paths["run_type"] == "user_formal"


def test_stage2_only_refuses_existing_outputs(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = tmp_path / "formal_outputs"
    existing = output_root / "synthetic_stage2_only" / "audit" / "02_cluster_results"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        script.build_stage2_only_run(
            _cfg(),
            stage1_dir=stage1,
            output_root=output_root,
            tag="audit",
            run_type="user_formal",
        )


def test_codex_test_output_must_stay_under_codex_results(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)

    with pytest.raises(ValueError, match="codex_test output_root"):
        script.build_stage2_only_run(
            _cfg(),
            stage1_dir=stage1,
            output_root=tmp_path / "outside_codex",
            tag="audit",
            run_type="codex_test",
        )


def test_codex_test_can_plan_under_codex_results_without_creating_stage3(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = PROJECT_ROOT / "local" / "results" / "codex" / "test_artifacts"

    cfg, paths = script.build_stage2_only_run(
        _cfg(),
        stage1_dir=stage1,
        output_root=output_root,
        tag="unit_only",
        run_type="codex_test",
    )

    assert script._is_relative_to(paths["output_root"], script.CODEX_RESULTS_ROOT)
    assert paths["cluster_dir"].name == "02_cluster_results"
    assert paths["result_dir"].name == "02_discovery_logs"
    assert "03_training_evaluation" not in str(paths["result_dir"])
    assert "04_model_artifacts" not in str(paths["result_dir"])
    assert cfg["stage2_only"]["stage1_dir"] == str(stage1.resolve())
