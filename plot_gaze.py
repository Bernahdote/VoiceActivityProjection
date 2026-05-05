"""
Plot the two gaze features from a .f.npz file.

Usage:
    uv run plot_gaze.py /path/to/file.f.npz
    uv run plot_gaze.py /path/to/file.f.npz --start 10 --end 30
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

GAZE_SLICE = slice(0, 2)
SRC_FPS = 30.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to .f.npz file")
    parser.add_argument("--start", type=float, default=None, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="End time in seconds")
    args = parser.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    gaze = z["features"][:, GAZE_SLICE]  # (T, 2)

    start_idx = int(args.start * SRC_FPS) if args.start is not None else 0
    end_idx = int(args.end * SRC_FPS) if args.end is not None else gaze.shape[0]
    gaze = gaze[start_idx:end_idx]

    t = np.arange(len(gaze)) / SRC_FPS
    if args.start:
        t += args.start

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(t, gaze[:, 0], label="Gaze X")
    ax.plot(t, gaze[:, 1], label="Gaze Y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gaze")
    ax.legend()
    ax.set_title(args.npz.split("/")[-1])
    plt.tight_layout()
    plt.savefig("gaze_plot.png", dpi=150)
    print("Saved to gaze_plot.png")


if __name__ == "__main__":
    main()
