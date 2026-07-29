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
        str(Path(tempfile.gettempdir()) / "semantic_split_multimodal_matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    train_rows = _read_numeric_rows(run_dir / "train_log.csv")
    validation_rows = _read_numeric_rows(run_dir / "validation_log.csv")
    if not train_rows:
        raise ValueError("train_log.csv contains no training rows.")
    if not validation_rows:
        raise ValueError("validation_log.csv contains no validation rows.")

    train_rounds = [int(row["round"]) for row in train_rows]
    train_loss = [float(row["loss"]) for row in train_rows]
    validation_success = [row for row in validation_rows if row["eval_status"] == "success"]
    if not validation_success:
        raise ValueError("validation_log.csv contains no successful validation rows.")

    validation_rounds = [int(row["round"]) for row in validation_success]
    validation_loss = [float(row["loss"]) for row in validation_success]
    validation_accuracy = [float(row["accuracy"]) for row in validation_success]
    validation_macro_f1 = [float(row["macro_f1"]) for row in validation_success]
    best_rows = [row for row in validation_success if int(row["is_best"]) == 1]
    best_round = int(best_rows[-1]["round"]) if best_rows else None
    stop_round = train_rounds[-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(train_rounds, train_loss, label="train loss", linewidth=1.5)
    axes[0].plot(
        validation_rounds,
        validation_loss,
        "o-",
        label="validation loss",
        linewidth=1.5,
    )
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        validation_rounds,
        validation_accuracy,
        "o-",
        label="validation accuracy",
    )
    axes[1].plot(
        validation_rounds,
        validation_macro_f1,
        "o-",
        label="validation macro-F1",
    )
    axes[1].set_xlabel("Global round")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    for axis in axes:
        if best_round is not None:
            axis.axvline(
                best_round,
                color="green",
                linestyle="--",
                alpha=0.7,
                label="_best",
            )
        axis.axvline(
            stop_round,
            color="red",
            linestyle=":",
            alpha=0.7,
            label="_stop",
        )

    fig.suptitle(
        f"Training curves (best round={best_round}, stop round={stop_round})"
    )
    fig.tight_layout()
    output_path = run_dir / "training_curves.png"
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
