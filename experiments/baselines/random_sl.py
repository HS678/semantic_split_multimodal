import sys

from experiments.msl.train import main as train_main, run_one

DEFAULT_METHOD = "randomsl"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--method" not in argv:
        argv = ["--method", DEFAULT_METHOD, *argv]
    return train_main(argv)


if __name__ == "__main__":
    main()
