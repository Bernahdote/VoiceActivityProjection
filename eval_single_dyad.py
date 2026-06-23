"""
Evaluate a checkpoint on clips of a single (dyad-disjoint) session, and
compare to the rest of the test set.

Usage:
  uv run python eval_single_dyad.py \\
      --ckpt /path/to/checkpoint.ckpt \\
      --test_pt /path/to/test_dir \\
      --test_csv /path/to/sliding.csv \\
      --session V00_S0534_I00000135 \\
      [--video_dim 364]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from vap.modules.lightning_module import VAPModule


EVENT_NAMES = ["hs", "ls", "sp"]


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _compute_bacc(preds, targets, threshold=0.5):
    p = (preds >= threshold).long()
    t = targets.long()
    acc = []
    for c in (0, 1):
        mask = t == c
        if mask.sum() == 0:
            acc.append(float("nan"))
        else:
            acc.append(float((p[mask] == c).float().mean()))
    if any(np.isnan(a) for a in acc):
        return float("nan")
    return (acc[0] + acc[1]) / 2.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_pt", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--session", required=True,
                        help="Session id to isolate (e.g. V00_S0534_I00000135)")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_config", default="vap/conf/stereo_home_dev.yaml")
    parser.add_argument("--video_dim", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.load(args.model_config)
    if args.video_dim is not None:
        cfg.module.model.video_dim = args.video_dim
    cfg.datamodule.test_path = args.test_pt
    cfg.datamodule.batch_size = args.batch_size
    cfg.datamodule.num_workers = args.num_workers

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    import pandas as pd
    df = pd.read_csv(args.test_csv)
    target_indices = set(df.index[df["session"].astype(str) == args.session].tolist())
    other_indices = set(range(len(df))) - target_indices
    print(f"{len(target_indices)} clips for session {args.session}")
    print(f"{len(other_indices)} other clips")
    if not target_indices:
        raise ValueError("Session has no clips in the test CSV.")

    module = VAPModule.load_from_checkpoint(args.ckpt, map_location="cpu", weights_only=False)
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    module.val_metric = instantiate(cfg.module.val_metric)
    module = module.to(device).eval()
    metric = module.val_metric

    # Cache events + per-clip loss per clip
    has_video = args.video_dim is not None and args.video_dim > 0
    cached = {"target": [], "other": []}
    clip_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            batch = _to_device(batch, device)
            if has_video and getattr(module.model, "video_dim", 0) > 0:
                out = module.model(
                    batch["waveform"],
                    video_features_a=batch["video_features_a"],
                    video_features_b=batch["video_features_b"],
                )
            else:
                out = module.model(batch["waveform"])
            probs = module.model.objective.get_probs(out["logits"])
            labels = module.model.extract_labels(batch["vad"])
            per_clip_loss = module.model.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).mean(dim=-1)
            bsz = batch["waveform"].shape[0]
            for b in range(bsz):
                metric.reset()
                probs_clip = {k: v[b:b+1] for k, v in probs.items()}
                vad_clip = batch["vad"][b:b+1]
                metric.update_batch(probs_clip, vad_clip)
                ev = {}
                for name in EVENT_NAMES:
                    if metric.preds[name]:
                        p = torch.cat(metric.preds[name]).detach().cpu()
                        t = torch.cat(metric.targets[name]).detach().cpu()
                        ev[name] = (p, t)
                bucket = "target" if clip_idx in target_indices else "other"
                cached[bucket].append({"events": ev, "loss": float(per_clip_loss[b])})
                clip_idx += 1

    print()
    for bucket in ("target", "other"):
        clips = cached[bucket]
        if not clips:
            continue
        label = f"Session {args.session}" if bucket == "target" else "All other clips"
        print(f"=== {label} ({len(clips)} clips) ===")
        for ev in EVENT_NAMES:
            preds, targets = [], []
            for c in clips:
                if ev in c["events"]:
                    p, t = c["events"][ev]
                    preds.append(p)
                    targets.append(t)
            if preds:
                bacc = _compute_bacc(torch.cat(preds), torch.cat(targets))
                print(f"  {ev.upper():<4} bAcc = {bacc:.4f}  ({sum(len(p) for p in preds)} predictions)")
        mean_loss = float(np.mean([c["loss"] for c in clips]))
        print(f"  LOSS     = {mean_loss:.4f}")
        print()


if __name__ == "__main__":
    main()
