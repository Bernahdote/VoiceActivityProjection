"""
Plot AU values over a short sequence with GT event regions highlighted.

Usage:
    uv run plot_au_events.py \
        --test_csv /path/to/test.csv \
        --sample_idx 0 \
        --duration 10
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from vap.data.datamodule import VAPDataModule
from vap.events.events import TurnTakingEvents, EventConfig

AU_NAMES = [
    "InnerBrowRaiser (AU1)",
    "OuterBrowRaiser (AU2)",
    "BrowLowerer (AU4)",
    "UpperLidRaiser (AU5)",
    "CheekRaiser (AU6)",
    "LidTightener (AU7)",
    "NoseWrinkler (AU9)",
    "UpperLipRaiser (AU10)",
    "LipCornerPuller (AU12)",
    "CheekPuffer (AU13)",
    "Dimpler (AU14)",
    "LipCornerDepressor (AU15)",
    "LowerLipDepressor (AU16)",
    "ChinRaiser (AU17)",
    "LipPuckerer (AU18)",
    "LipStretcher (AU20)",
    "LipFunneler (AU22)",
    "LipTightener (AU23)",
    "LipPressor (AU24)",
    "LipsParts (AU25)",
    "JawDrop (AU26)",
    "LipSuck (AU28)",
    "JawSideways (AU30)",
    "EyesClosed (AU43)",
]

EVENT_COLORS = {
    "shift":               "red",
    "hold":                "blue",
    "long":                "green",
    "short":               "orange",
    "pred_shift":          "salmon",
    "pred_hold":           "cornflowerblue",
    "pred_backchannel":    "mediumpurple",
    "pred_backchannel_neg":"plum",
}


def draw_events(ax, events: dict, b: int, n_frames: int, alpha: float = 0.25):
    for event_name, color in EVENT_COLORS.items():
        key = "pred_shift_neg" if event_name == "pred_hold" else event_name
        if key not in events:
            continue
        for start, end, speaker in events[key][b]:
            start = min(start, n_frames)
            end = min(end, n_frames)
            if start >= end:
                continue
            ax.axvspan(start, end, color=color, alpha=alpha, linewidth=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Seconds to plot (from start of clip)")
    parser.add_argument("--output", default="plot_au_events.png")
    args = parser.parse_args()

    dm = VAPDataModule(
        train_path=args.test_csv,
        val_path=args.test_csv,
        test_path=args.test_csv,
        batch_size=1,
        num_workers=0,
        video_feature_groups=["fauv"],
    )
    dm.prepare_data()
    dm.setup("test")
    loader = dm.test_dataloader()

    # Grab the requested sample
    for i, batch in enumerate(loader):
        if i == args.sample_idx:
            break

    frame_hz = 50
    n_frames = int(args.duration * frame_hz)

    vad    = batch["vad"][0, :n_frames].numpy()          # (T, 2)
    feat_a = batch["video_features_a"][0, :n_frames].numpy()  # (T, 24)
    feat_b = batch["video_features_b"][0, :n_frames].numpy()  # (T, 24)

    # Trim vad tensor for event extraction (must be 3D)
    vad_t = batch["vad"][:, :n_frames]
    events = TurnTakingEvents(EventConfig())(vad_t)

    t = np.arange(n_frames) / frame_hz

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 8),
        gridspec_kw={"height_ratios": [1, 2, 2]},
        sharex=True,
    )

    # ── VAD ──────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(t, vad[:, 0], alpha=0.7, label="Speaker A", color="steelblue")
    ax.fill_between(t, -vad[:, 1], alpha=0.7, label="Speaker B", color="tomato")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("VAD")
    ax.set_ylim(-1.2, 1.2)
    ax.legend(loc="upper right", fontsize=7)
    draw_events(ax, events, b=0, n_frames=n_frames)

    AU26_IDX = 20

    # ── AU26 JawDrop: Speaker A ───────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, feat_a[:, AU26_IDX], color="steelblue", linewidth=0.8)
    ax.set_ylabel("JawDrop (AU26)\nSpeaker A")
    ax.set_ylim(bottom=0)
    draw_events(ax, events, b=0, n_frames=n_frames)

    # ── AU26 JawDrop: Speaker B ───────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, feat_b[:, AU26_IDX], color="tomato", linewidth=0.8)
    ax.set_ylabel("JawDrop (AU26)\nSpeaker B")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(bottom=0)
    draw_events(ax, events, b=0, n_frames=n_frames)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, alpha=0.5, label=name)
        for name, c in EVENT_COLORS.items()
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(f"AU values with GT events — sample {args.sample_idx}", fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
