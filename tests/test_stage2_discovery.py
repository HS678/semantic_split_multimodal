import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "stage2_discovery.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stage2_discovery", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stage1_dir(tmp_path):
    path = tmp_path / "stage1" / "acc_1clients_gyro_1clients"
    (path / "train_clients").mkdir(parents=True)
    return path


def _cfg():
    return {
        "seed": 42,
        "dataset": {"type": "synthetic_stage2"},
        "cluster": {
            "method": "adaptive_isodata",
            "known_k": None,
            "adaptive": {"seeds": [11, 23], "max_iter": 2},
        },
    }


def test_stage2_builds_isolated_output_paths_without_touching_stage1(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = tmp_path / "formal_outputs"

    cfg, paths = script.build_stage2_run(
        _cfg(),
        stage1_dir=stage1,
        output_root=output_root,
    )

    assert Path(cfg["partition"]["output_dir"]) == stage1.resolve()
    expected_run = output_root.resolve() / "synthetic_stage2" / stage1.name / "adaptive_isodata"
    assert Path(cfg["cluster"]["output_dir"]) == expected_run
    assert Path(cfg["result"]["output_dir"]) == expected_run
    assert not any(stage1.parent.glob("synthetic_stage2*"))
    assert paths["cluster_method"] == "adaptive_isodata"


def test_stage2_refuses_existing_outputs(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = tmp_path / "formal_outputs"
    existing = output_root / "synthetic_stage2" / stage1.name / "adaptive_isodata"
    existing.mkdir(parents=True)

    with pytest.raises(FileExistsError):
        script.build_stage2_run(
            _cfg(),
            stage1_dir=stage1,
            output_root=output_root,
        )


def test_stage2_optional_run_name_preserves_other_config_outputs(tmp_path):
    script = _load_script()
    stage1 = _stage1_dir(tmp_path)
    output_root = tmp_path / "formal_outputs"
    cfg = _cfg()
    cfg["stage2"] = {"run_name": "temporal_bigru_cls_v1"}

    resolved, paths = script.build_stage2_run(
        cfg,
        stage1_dir=stage1,
        output_root=output_root,
    )
    expected = (
        output_root.resolve()
        / "synthetic_stage2"
        / stage1.name
        / "adaptive_isodata"
        / "temporal_bigru_cls_v1"
    )
    assert paths["cluster_dir"] == expected
    assert paths["run_name"] == "temporal_bigru_cls_v1"
    assert resolved["stage2"]["run_name"] == "temporal_bigru_cls_v1"


def test_stage2_cli_rejects_removed_run_type_option():
    script = _load_script()

    with pytest.raises(SystemExit):
        script.parse_args(
            [
                "--config",
                "configs/uci_har.config",
                "--stage1-dir",
                "stage1",
                "--run-type",
                "user_formal",
            ]
        )
