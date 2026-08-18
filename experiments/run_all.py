import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_all_discovery
from experiments.msl import run_all as run_all_training


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run discovery comparison and all training methods.")
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--methods", nargs="*")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--global-rounds", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args(argv)


def _shared_args(args) -> list[str]:
    out = ["--results-root", args.results_root]
    if args.datasets:
        out.extend(["--datasets", *args.datasets])
    if args.seeds:
        out.extend(["--seeds", *(str(seed) for seed in args.seeds)])
    return out


def _training_args(args) -> list[str]:
    out = _shared_args(args)
    if args.methods:
        out.extend(["--methods", *args.methods])
    if args.device:
        out.extend(["--device", args.device])
    if args.global_rounds is not None:
        out.extend(["--global-rounds", str(args.global_rounds)])
    if args.retry_failed:
        out.append("--retry-failed")
    if args.require_cuda:
        out.append("--require-cuda")
    return out


def main(argv=None):
    args = parse_args(argv)
    if not args.skip_discovery:
        run_all_discovery.main(_shared_args(args))
    if not args.skip_training:
        run_all_training.main(_training_args(args))


if __name__ == "__main__":
    main()
