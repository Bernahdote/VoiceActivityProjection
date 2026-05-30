"""
Analyse the average value of a specific AU feature index across turn-taking events.

Usage:
    uv run au_event_analysis.py \
        --test_csv /path/to/test.csv \
        --batch_size 20 \
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
    "pred_backchannel",
    "pred_backchannel_neg",
]


AU26_IDX = 20  # JawDrop (AU26) is at index 20 in the fauv array


def collect_au_values(
    events: dict,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    batch_size: int,
    acc_actor: dict[str, list],
    acc_other: dict[str, list],
) -> None:
    """Collect AU26 (JawDrop) at the single frame right before the event boundary."""
    T = feat_a.shape[1]  # number of valid feature frames
    for event_name in EVENT_NAMES:
        key = "pred_shift_neg" if event_name == "pred_hold" else event_name
        if key not in events:
            continue
        for b in range(batch_size):
            for start, end, speaker in events[key][b]:
                if end <= start:
                    continue
                last_idx = min(end - 1, T - 1)  # clamp to valid frame range
                if last_idx < 0:
                    continue
                feats = [feat_a, feat_b]
                actor_val = float(feats[speaker][b, last_idx, AU26_IDX])
                other_val = float(feats[1 - speaker][b, last_idx, AU26_IDX])
                acc_actor[event_name].append(actor_val)
                acc_other[event_name].append(other_val)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
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

    event_extractor = TurnTakingEvents(EventConfig())
    acc_actor: dict[str, list] = defaultdict(list)
    acc_other: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            vad = batch["vad"]
            feat_a = batch["video_features_a"]
            feat_b = batch["video_features_b"]
            bsz = vad.shape[0]

            events = event_extractor(vad)
            collect_au_values(events, feat_a, feat_b, bsz, acc_actor, acc_other)

    print(f"\n=== AU26 (JawDrop) at the frame right before the event ===")
    header = (
        f"{'Event':<22}  {'Actor mean':>12}  {'Actor std':>10}"
        f"  {'Other mean':>12}  {'Other std':>10}  {'N events':>10}"
    )
    print(header)
    print("-" * len(header))
    for event_name in EVENT_NAMES:
        a = acc_actor[event_name]
        o = acc_other[event_name]
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
