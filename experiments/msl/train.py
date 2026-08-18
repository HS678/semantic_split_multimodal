import sys

from experiments.training import main as training_main, run_one


DEFAULT_METHOD = "ours"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" in argv:
        raise SystemExit("experiments/msl/train.py fixes method=ours; use experiments/training.py for explicit methods.")
    argv = ["--method", DEFAULT_METHOD, *argv]
    return training_main(argv)


if __name__ == "__main__":
    main()
