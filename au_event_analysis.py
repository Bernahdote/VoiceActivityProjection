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
TAIL_FRAMES = 10  # last frames of the event window (~200 ms at 50 Hz)


def collect_au_values(
    events: dict,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    batch_size: int,
    acc_actor: dict[str, dict[str, list]],
    acc_other: dict[str, dict[str, list]],
) -> None:
    """Collect summary statistics of AU26 (JawDrop) per event window."""
    for event_name in EVENT_NAMES:
        key = "pred_shift_neg" if event_name == "pred_hold" else event_name
        if key not in events:
            continue
        for b in range(batch_size):
            for start, end, speaker in events[key][b]:
                feats = [feat_a, feat_b]
                actor_chunk = feats[speaker][b, start:end, AU26_IDX].cpu().numpy()
                other_chunk = feats[1 - speaker][b, start:end, AU26_IDX].cpu().numpy()
                if actor_chunk.size == 0:
                    continue
                tail_actor = actor_chunk[-TAIL_FRAMES:]
                tail_other = other_chunk[-TAIL_FRAMES:]
                acc_actor[event_name]["mean"].append(float(np.mean(actor_chunk)))
                acc_actor[event_name]["max"].append(float(np.max(actor_chunk)))
                acc_actor[event_name]["tail_mean"].append(float(np.mean(tail_actor)))
                acc_other[event_name]["mean"].append(float(np.mean(other_chunk)))
                acc_other[event_name]["max"].append(float(np.max(other_chunk)))
                acc_other[event_name]["tail_mean"].append(float(np.mean(tail_other)))


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
    acc_actor: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    acc_other: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            vad = batch["vad"]
            feat_a = batch["video_features_a"]
            feat_b = batch["video_features_b"]
            bsz = vad.shape[0]

            events = event_extractor(vad)
            collect_au_values(events, feat_a, feat_b, bsz, acc_actor, acc_other)

    print(f"\n=== AU26 (JawDrop) per event (last {TAIL_FRAMES} frames ≈ {TAIL_FRAMES*20} ms) ===")
    header = (
        f"{'Event':<22}  {'A mean':>8}  {'A max':>8}  {'A tail':>8}"
        f"  {'O mean':>8}  {'O max':>8}  {'O tail':>8}  {'N':>8}"
    )
    print(header)
    print("-" * len(header))
    for event_name in EVENT_NAMES:
        a = acc_actor[event_name]
        o = acc_other[event_name]
        if not a.get("mean"):
            continue
        n = len(a["mean"])
        print(
            f"{event_name:<22}"
            f"  {np.mean(a['mean']):>8.4f}  {np.mean(a['max']):>8.4f}  {np.mean(a['tail_mean']):>8.4f}"
            f"  {np.mean(o['mean']):>8.4f}  {np.mean(o['max']):>8.4f}  {np.mean(o['tail_mean']):>8.4f}"
            f"  {n:>8d}"
        )


if __name__ == "__main__":
    main()
