import argparse
import csv
import os
from pathlib import Path
import tempfile


def _read_numeric_rows(path: Path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_training_curves(run_dir: Path):
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "MSL_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train_rows = _read_numeric_rows(run_dir / "train_log.csv")
    if not train_rows:
        raise ValueError("train_log.csv contains no training rows.")

    train_rounds = [int(row["round"]) for row in train_rows]
    train_loss = [float(row["loss"]) for row in train_rows]
    stop_round = train_rounds[-1]

    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    axes.plot(train_rounds, train_loss, label="train loss", linewidth=1.5)
    axes.set_xlabel("Global round")
    axes.set_ylabel("Loss")
    axes.grid(alpha=0.25)
    axes.legend()
    axes.axvline(
        stop_round,
        color="red",
        linestyle=":",
        alpha=0.7,
        label="_stop",
    )
    fig.suptitle(f"Training curves (fixed rounds={stop_round})")
    fig.tight_layout()
    output_path = run_dir / "training_curves.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot Stage3 training loss curve from a completed run."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Stage3 run directory containing train_log.csv and validation_log.csv",
    )
    args = parser.parse_args(argv)
    output_path = write_training_curves(Path(args.run_dir))
    print(f"training_curves={output_path}")


if __name__ == "__main__":
    main()
