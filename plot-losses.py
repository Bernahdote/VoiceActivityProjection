"""
Plot validation-loss curves for all feature-set variants.

Expected CSV files (download from WandB, one per run):
  Losses/
    Baseline-loss.csv
    All-loss.csv
    Body-loss.csv
    Gaze-loss.csv
    FAUV-loss.csv
    Body+Gaze-loss.csv
    Body+FAU-loss.csv
    FAU+Gaze-loss.csv

Each CSV is expected to have two columns: 'trainer/global_step' and a
'<run_name> - val_loss' column. The script auto-detects the val_loss column.
Files that don't exist are silently skipped, so you can plot whichever subset
you've downloaded so far.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd

LOSSES_DIR = "/Users/willemberner/Desktop/Exjobb/Losses"

# (file_stub, label, color)
RUNS = [
    ("Baseline",   "Baseline",   "red"),
    ("M5",         "All features",       "purple"),
    ("Body",       "Body",                    "blue"),
    ("Gaze",       "Gaze",      "orange"),
    ("FAU",       "FAU",                     "green"),
    ("Body+Gaze",  "Body + Gaze",             "teal"),
    ("Body+FAU",   "Body + FAU",              "brown"),
    ("FAU+Gaze",   "FAU + Gaze",              "magenta"),
]


def _val_loss_col(df: pd.DataFrame) -> str | None:
    """Return the column that holds the val_loss values."""
    for c in df.columns:
        if "val_loss" in c and "MIN" not in c.upper() and "MAX" not in c.upper():
            return c
    return None


def main():
    # First pass: collect available runs
    loaded = []
    for stub, label, color in RUNS:
        path = os.path.join(LOSSES_DIR, f"{stub}-loss.csv")
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        col = _val_loss_col(df)
        if col is None:
            print(f"[skip] {path} has no val_loss column. Columns: {list(df.columns)}")
            continue
        loaded.append((label, color, df, col))

    if not loaded:
        raise RuntimeError(f"No usable loss CSVs found in {LOSSES_DIR}")

    # Clip every curve to the step where the shortest run ended.
    max_step = min(df["trainer/global_step"].max() for _, _, df, _ in loaded)
    print(f"Trimming all curves to step <= {max_step}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for label, color, df, col in loaded:
        df = df[df["trainer/global_step"] <= max_step]
        ax.plot(df["trainer/global_step"], df[col], label=label, color=color, linewidth=1.5)

    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
