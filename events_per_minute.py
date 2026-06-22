"""
Count shifts and holds per minute in the test set, split by domain
(naturalistic vs improvised).

Approach:
  - Iterate over the sliding-window .pt files (each contains the VAD for a 20s clip).
  - Extract turn-taking events with TurnTakingEvents.
  - Tag each clip's domain via the CSV's audio_path_a substring.
  - Sum events and time per domain, then compute events / minute.

Usage:
  uv run python events_per_minute.py \\
      --test_pt /mnt/sdb/willem/datasets/preprocessed/test_3_all_200 \\
      --test_csv /mnt/sdb/willem/datasets/splits_200/test_sliding.csv
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from vap.events.events import TurnTakingEvents, EventConfig


CLIP_DURATION_S = 20.0
FRAME_HZ = 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pt", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    df = pd.read_csv(args.test_csv)
    df["domain"] = df["audio_path_a"].str.lower().apply(
        lambda p: "improvised" if "improvised" in p else "naturalistic"
    )

    pt_files = sorted(Path(args.test_pt).glob("*.pt"))
    if len(pt_files) != len(df):
        print(f"[warn] {len(pt_files)} .pt files vs {len(df)} CSV rows; "
              "assuming index-based correspondence.")

    extractor = TurnTakingEvents(EventConfig(equal_hold_shift=False))

    # Counters per domain
    counts = defaultdict(lambda: defaultdict(int))   # counts[domain][event_name]
    n_clips = defaultdict(int)

    with torch.no_grad():
        # process in batches of CLIPS by loading VAD into a batched tensor
        for batch_start in tqdm(range(0, len(pt_files), args.batch_size), desc="Counting"):
            batch_end = min(batch_start + args.batch_size, len(pt_files))
            vads = []
            domains = []
            for i in range(batch_start, batch_end):
                d = torch.load(pt_files[i], weights_only=False)
                vads.append(d["vad"])
                domains.append(df["domain"].iloc[i])
            vad = torch.stack(vads)  # (B, T, 2)
            events = extractor(vad)
            for b, dom in enumerate(domains):
                n_clips[dom] += 1
                for name in ("shift", "hold"):
                    counts[dom][name] += len(events[name][b])

    print("\n=== Shifts and Holds per minute (test set) ===")
    print(f"{'Domain':<14}  {'Clips':>6}  {'Minutes':>9}  "
          f"{'Shifts':>7}  {'Sh/min':>8}  {'Holds':>7}  {'H/min':>8}")
    for dom in ("improvised", "naturalistic"):
        clips = n_clips[dom]
        minutes = clips * CLIP_DURATION_S / 60
        sh = counts[dom]["shift"]
        ho = counts[dom]["hold"]
        sh_per_min = sh / minutes if minutes else 0
        ho_per_min = ho / minutes if minutes else 0
        print(f"{dom:<14}  {clips:>6}  {minutes:>9.1f}  "
              f"{sh:>7}  {sh_per_min:>8.2f}  {ho:>7}  {ho_per_min:>8.2f}")


if __name__ == "__main__":
    main()
