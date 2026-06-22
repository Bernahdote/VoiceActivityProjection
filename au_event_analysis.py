"""
Analyse the average value of selected FAUs at the frame right before each
turn-taking event, for the actor (event speaker) and the other speaker.

Selected FAUs: AU4 (BrowLowerer), AU10 (UpperLipRaiser), AU12 (LipCornerPull),
AU26 (JawDrop), AU43 (EyesClosed).

Usage:
    uv run python au_event_analysis.py \\
        --test_csv /path/to/test.csv (or .pt directory) \\
        --batch_size 20 \\
        --num_workers 4
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from vap.data.datamodule import VAPDataModule
from vap.events.events import TurnTakingEvents, EventConfig


EVENT_NAMES = [
    "shift",
    "hold",
    "long",
    "short",
    "pred_shift",
    "pred_hold",
]

# AU index in the 24-dim fauv array (FAU value features)
AU_INDICES = {
    "AU4 (BrowLowerer)": 2,
    "AU10 (UpperLipRaiser)": 7,
    "AU12 (LipCornerPull)": 8,
    "AU26 (JawDrop)": 20,
    "AU43 (EyesClosed)": 23,
}

# Number of frames immediately before each event window to average over (1 = single frame just before)
PRE_FRAMES = 1



def collect_au_values(
    events: dict,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    batch_size: int,
    acc_p1: dict[str, dict[str, list[float]]],
    acc_p2: dict[str, dict[str, list[float]]],
) -> None:
    """For each event, average each selected AU value across the PRE_FRAMES
    frames immediately BEFORE the event window starts, for both speakers.
    """
    T = feat_a.shape[1]
    for event_name in EVENT_NAMES:
        key = "pred_shift_neg" if event_name == "pred_hold" else event_name
        if key not in events:
            continue
        for b in range(batch_size):
            for start, end, speaker in events[key][b]:
                pre_end = min(max(start, 0), T)
                pre_start = max(pre_end - PRE_FRAMES, 0)
                if pre_end <= pre_start:
                    continue
                feats = [feat_a, feat_b]
                for au_name, idx in AU_INDICES.items():
                    p1_val = float(feats[speaker][b, pre_start:pre_end, idx].mean())
                    p2_val = float(feats[1 - speaker][b, pre_start:pre_end, idx].mean())
                    acc_p1[event_name][au_name].append(p1_val)
                    acc_p2[event_name][au_name].append(p2_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True,
                        help="Test CSV or .pt directory (datamodule auto-detects).")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dm = VAPDataModule(
        train_path=args.test_csv,
        val_path=args.test_csv,
        test_path=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        video_feature_groups=["fauv"],
    )
    dm.prepare_data()
    dm.setup("test")
    loader = dm.test_dataloader()

    event_extractor = TurnTakingEvents(EventConfig(equal_hold_shift=False))
    acc_p1: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    acc_p2: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            vad = batch["vad"]
            feat_a = batch["video_features_a"]
            feat_b = batch["video_features_b"]
            bsz = vad.shape[0]

            events = event_extractor(vad)
            collect_au_values(events, feat_a, feat_b, bsz, acc_p1, acc_p2)

    # Print one section per AU
    for au_name in AU_INDICES:
        print(f"\n=== {au_name} ===")
        header = (
            f"{'Event':<22}  {'P1 mean':>12}  {'P1 std':>10}"
            f"  {'P2 mean':>12}  {'P2 std':>10}  {'N events':>10}"
        )
        print(header)
        print("-" * len(header))
        for event_name in EVENT_NAMES:
            a = acc_p1[event_name][au_name]
            o = acc_p2[event_name][au_name]
            if not a:
                continue
            av = np.asarray(a)
            ov = np.asarray(o)
            print(
                f"{event_name:<22}"
                f"  {np.mean(av):>12.4f}  {np.std(av):>10.4f}"
                f"  {np.mean(ov):>12.4f}  {np.std(ov):>10.4f}"
                f"  {len(av):>10d}"
            )


if __name__ == "__main__":
    main()
