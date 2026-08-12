"""randomSL baseline Stage 3 entry point.

Runs the randomSL baseline (random-scheduling Split Learning) on top of the
frozen Stage1 partition and Stage2 cluster assignments produced by the MSL
mainline. The run directory layout, input audit, metadata, completion status
and summary format are identical to the mainline ``stage3_train.py`` so that
``summarize_results.py --results-root local/results_baseline/randomSL`` works
unchanged.

The mainline script is imported (not executed) so path building, auditing and
metadata helpers are reused without duplicating them.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseline.randomSL.training import run_random_sl_stage3_split_training
from MSL.evaluation.plot_training_curves import write_training_curves
from MSL.utils.config import apply_experiment_overrides, load_config, normalize_experiment_config, save_config_artifacts
from MSL.utils.device import select_device
from MSL.utils.results import resolve_stage_paths
from MSL.utils.seed import set_seed


def _load_stage3_script():
    spec = importlib.util.spec_from_file_location(
        "stage3_train",
        ROOT / "scripts" / "stage3_train.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _baseline_metadata(stage3, args, cfg, paths, audit, status, failure_reason, start, end, metrics=None):
    metadata = stage3._metadata(
        args,
        cfg,
        paths,
        audit,
        status,
        failure_reason,
        start,
        end,
        metrics=metrics,
    )
    metadata["baseline"] = "randomSL"
    metadata["method"] = "random_sl"
    metadata["scheduler"] = "random"
    metadata["training_mode"] = "random_sl_split_learning"
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage 3 baseline: random-scheduling Split Learning (randomSL)."
    )
    parser.add_argument("--config", required=True, help="Path to INI-style .config file")
    parser.add_argument("--fold", type=int, help="Override dataset.split_protocol from the dataset fold template.")
    parser.add_argument("--split-protocol", help="Override dataset.split_protocol directly.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the Stage3 experiment seed in memory; does not modify the source .config or affect Stage1/Stage2",
    )
    parser.add_argument(
        "--fusion-training-objective",
        choices=["label_random_ce", "mmbind_weighted_contrastive"],
        help="Override fusion.training_objective in memory and record it in resolved_config.config",
    )
    parser.add_argument("--stage1-dir", help="Optional override for stage3.stage1_dir")
    parser.add_argument("--stage2-dir", help="Optional override for stage3.stage2_dir")
    parser.add_argument("--output-root", help="Optional override for stage3.output_root")
    parser.add_argument(
        "--attempt",
        type=int,
        help="Optional override for stage3.attempt; the run fails instead of overwriting an existing attempt",
    )
    return parser.parse_args(argv)


def main(argv=None):
    stage3 = _load_stage3_script()
    args = parse_args(argv)
    cfg = normalize_experiment_config(load_config(args.config))
    cfg = apply_experiment_overrides(cfg, fold=args.fold, split_protocol=args.split_protocol)
    stage3_cfg = cfg.get("stage3", {})
    stage1_dir = args.stage1_dir or stage3_cfg.get("stage1_dir")
    stage2_dir = args.stage2_dir or stage3_cfg.get("stage2_dir")
    if not stage1_dir or not stage2_dir:
        resolved = resolve_stage_paths(cfg, ROOT)
        stage1_dir = stage1_dir or resolved["stage1_dir"]
        stage2_dir = stage2_dir or resolved["stage2_dir"]
    output_root = args.output_root or stage3_cfg.get("output_root")
    if not output_root:
        output_root = resolve_stage_paths(cfg, ROOT)["output_dir"]
    if not stage1_dir or not stage2_dir:
        raise ValueError(
            "Set stage3.stage1_dir and stage3.stage2_dir in the .config file or pass CLI overrides."
        )
    attempt = args.attempt if args.attempt is not None else int(stage3_cfg.get("attempt", 1))
    resolved_seed = int(args.seed) if args.seed is not None else int(cfg.get("seed", 42))
    cfg = {**cfg, "seed": resolved_seed}
    if args.fusion_training_objective is not None:
        cfg["fusion"] = {
            **cfg.get("fusion", {}),
            "training_objective": args.fusion_training_objective,
        }
    # attempt 自动递增：同 loss 目录已存在时自动尝试下一个 attempt，避免覆盖旧结果。
    run_cfg = None
    paths = None
    for candidate_attempt in range(attempt, attempt + 100):
        try:
            run_cfg, paths = stage3.build_stage3_run(
                cfg,
                stage1_dir=stage1_dir,
                stage2_dir=stage2_dir,
                output_root=output_root,
                attempt=candidate_attempt,
            )
            attempt = candidate_attempt
            break
        except FileExistsError:
            continue
    if run_cfg is None or paths is None:
        raise FileExistsError(f"Too many existing Stage3 attempt directories under {output_root}.")
    audit = stage3.audit_stage3_inputs(run_cfg, paths["stage1_dir"], paths["stage2_dir"])

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    save_config_artifacts(args.config, run_cfg, paths["run_dir"])

    start = stage3._utc_now()
    stage3._metadata.start_monotonic = time.time()
    try:
        set_seed(int(run_cfg.get("seed", 42)))
        device = select_device(run_cfg.get("device", "auto"))
        metrics = run_random_sl_stage3_split_training(run_cfg, ROOT, device)
        write_training_curves(paths["run_dir"])
    except Exception as exc:
        end = stage3._utc_now()
        stage3._write_json(
            paths["metadata"],
            _baseline_metadata(stage3, args, run_cfg, paths, audit, "failed", str(exc), start, end),
        )
        raise

    end = stage3._utc_now()
    status, failure_reason = stage3._formal_completion_status(metrics, paths)
    stage3._write_json(
        paths["metadata"],
        _baseline_metadata(
            stage3,
            args,
            run_cfg,
            paths,
            audit,
            status,
            failure_reason,
            start,
            end,
            metrics=metrics,
        ),
    )
    if status != "success":
        raise RuntimeError(f"Stage3 run did not complete successfully: {failure_reason}")
    print("Stage 3 (randomSL) finished.")
    print(f"stage1_dir={paths['stage1_dir']}")
    print(f"stage2_dir={paths['stage2_dir']}")
    print(f"run_dir={paths['run_dir']}")
    print(f"final_metrics={metrics}")


if __name__ == "__main__":
    main()
