"""
Analyse the average value of a specific AU feature index across turn-taking events.

Usage:
    uv run au_event_analysis.py \
        --test_csv /path/to/test.csv \
        --au_idx 13 \
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
    "pred_shift_neg",
    "pred_backchannel",
    "pred_backchannel_neg",
]


def collect_au_values(
    events: dict,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    au_idx: int,
    batch_size: int,
    accumulator: dict[str, list],
) -> None:
    """For each event in the batch, collect AU values from the relevant speaker."""
    for event_name in EVENT_NAMES:
        if event_name not in events:
            continue
        for b in range(batch_size):
            for start, end, speaker in events[event_name][b]:
                feat = feat_a if speaker == 0 else feat_b
                vals = feat[b, start:end, au_idx].cpu().numpy()
                accumulator[event_name].append(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--au_idx", type=int, default=13)
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
    accumulator: dict[str, list] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            vad = batch["vad"]
            feat_a = batch["video_features_a"]
            feat_b = batch["video_features_b"]
            bsz = vad.shape[0]

            events = event_extractor(vad)
            collect_au_values(events, feat_a, feat_b, args.au_idx, bsz, accumulator)

    print(f"\nAU index {args.au_idx}  |  features: fauv\n")
    print(f"{'Event':<25}  {'N frames':>10}  {'Mean':>10}  {'Std':>10}")
    print("-" * 60)
    for event_name in EVENT_NAMES:
        vals = accumulator[event_name]
        if not vals:
            print(f"{event_name:<25}  {'—':>10}  {'—':>10}  {'—':>10}")
            continue
        all_vals = np.concatenate(vals)
        print(
            f"{event_name:<25}  {len(all_vals):>10}  {np.mean(all_vals):>10.4f}  {np.std(all_vals):>10.4f}"
        )


if __name__ == "__main__":
    main()
