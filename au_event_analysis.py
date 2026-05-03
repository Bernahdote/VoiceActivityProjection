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


def collect_au_values(
    events: dict,
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    batch_size: int,
    acc_actor: dict[str, dict[int, list]],
    acc_other: dict[str, dict[int, list]],
    n_features: int,
) -> None:
    """Collect AU values for both the event speaker and the other speaker, all features."""
    for event_name in EVENT_NAMES:
        key = "pred_shift_neg" if event_name == "pred_hold" else event_name
        if key not in events:
            continue
        for b in range(batch_size):
            for start, end, speaker in events[key][b]:
                feats = [feat_a, feat_b]
                actor_chunk = feats[speaker][b, start:end, :n_features].cpu().numpy()  # (T, n_features)
                other_chunk = feats[1 - speaker][b, start:end, :n_features].cpu().numpy()
                for au_idx in range(n_features):
                    acc_actor[event_name][au_idx].append(actor_chunk[:, au_idx])
                    acc_other[event_name][au_idx].append(other_chunk[:, au_idx])


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
    n_features = 24  # fauv has 24 AU dimensions
    acc_actor: dict[str, dict[int, list]] = {e: defaultdict(list) for e in EVENT_NAMES}
    acc_other: dict[str, dict[int, list]] = {e: defaultdict(list) for e in EVENT_NAMES}

    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing"):
            vad = batch["vad"]
            feat_a = batch["video_features_a"]
            feat_b = batch["video_features_b"]
            bsz = vad.shape[0]

            events = event_extractor(vad)
            collect_au_values(events, feat_a, feat_b, bsz, acc_actor, acc_other, n_features)

    AU_NAMES = [
        "InnerBrowRaiser",   # AU1
        "OuterBrowRaiser",   # AU2
        "BrowLowerer",       # AU4
        "UpperLidRaiser",    # AU5
        "CheekRaiser",       # AU6
        "LidTightener",      # AU7
        "NoseWrinkler",      # AU9
        "UpperLipRaiser",    # AU10
        "LipCornerPuller",   # AU12
        "CheekPuffer",       # AU13
        "Dimpler",           # AU14
        "LipCornerDepressor",# AU15
        "LowerLipDepressor", # AU16
        "ChinRaiser",        # AU17
        "LipPuckerer",       # AU18
        "LipStretcher",      # AU20
        "LipFunneler",       # AU22
        "LipTightener",      # AU23
        "LipPressor",        # AU24
        "LipsParts",         # AU25
        "JawDrop",           # AU26
        "LipSuck",           # AU28
        "JawSideways",       # AU30
        "EyesClosed",        # AU43
    ]

    for event_name in EVENT_NAMES:
        print(f"\n=== {event_name} ===")
        print(f"{'AU':<22}  {'Actor mean':>12}  {'Actor std':>10}  {'Other mean':>12}  {'Other std':>10}")
        print("-" * 74)
        for au_idx in range(n_features):
            a = acc_actor[event_name][au_idx]
            o = acc_other[event_name][au_idx]
            if not a:
                continue
            av = np.concatenate(a)
            ov = np.concatenate(o)
            print(
                f"{AU_NAMES[au_idx]:<22}"
                f"  {np.mean(av):>12.4f}  {np.std(av):>10.4f}"
                f"  {np.mean(ov):>12.4f}  {np.std(ov):>10.4f}"
            )


if __name__ == "__main__":
    main()
