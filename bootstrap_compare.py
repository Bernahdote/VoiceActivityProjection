"""
Bootstrap 95% confidence intervals for a single model's metrics.

Strategy:
  - Run inference once, then run the metric.update_batch ONCE per clip and
    snapshot the resulting (pred, target) tensors per clip per event type.
  - Bootstrap by sampling sessions, concatenating cached tensors, threshold +
    compute accuracy/F1 directly (no re-running event extraction).

Usage:
  uv run python bootstrap_compare.py \\
      --ckpt /path/to/checkpoint.ckpt \\
      --test_pt /path/to/test_dir \\
      --test_csv /path/to/sliding.csv \\
      [--video_dim 306] [--n_boot 1000]
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
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


def _load_module(ckpt_path: Path, cfg, device):
    module = VAPModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    module.val_metric = instantiate(cfg.module.val_metric)
    return module.to(device).eval()


def _run_inference_and_cache_events(module, loader, has_video: bool, sessions: list[str]):
    """For each clip: forward pass, then run metric.update_batch ONCE and snapshot
    the resulting per-event (pred, target) tensors. Returns list of per-clip dicts.
    """
    metric = module.val_metric
    cached: list[dict] = []
    clip_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference + event cache"):
            batch = _to_device(batch, module.device)
            if has_video and getattr(module.model, "video_dim", 0) > 0:
                out = module.model(
                    batch["waveform"],
                    video_features_a=batch["video_features_a"],
                    video_features_b=batch["video_features_b"],
                )
            else:
                out = module.model(batch["waveform"])
            probs = module.model.objective.get_probs(out["logits"])
            bsz = batch["waveform"].shape[0]
            for b in range(bsz):
                # Reset and run metric on single clip to extract its events
                metric.reset()
                probs_clip = {k: v[b:b+1] for k, v in probs.items()}
                vad_clip = batch["vad"][b:b+1]
                metric.update_batch(probs_clip, vad_clip)
                # Snapshot per-event preds and targets for this clip
                clip_events = {}
                for ev in EVENT_NAMES:
                    if metric.preds[ev]:
                        p = torch.cat(metric.preds[ev]).detach().cpu()
                        t = torch.cat(metric.targets[ev]).detach().cpu()
                        clip_events[ev] = (p, t)
                cached.append({
                    "session": sessions[clip_idx] if clip_idx < len(sessions) else str(clip_idx),
                    "events": clip_events,
                })
                clip_idx += 1
    metric.reset()
    return cached


def _compute_bacc(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute balanced accuracy."""
    p = (preds >= threshold).long()
    t = targets.long()
    acc_per_class = []
    for c in (0, 1):
        mask = t == c
        if mask.sum() == 0:
            acc_per_class.append(0.0)
        else:
            acc_per_class.append(float((p[mask] == c).float().mean()))
    return (acc_per_class[0] + acc_per_class[1]) / 2.0


def _metrics_from_clips(clips: list[dict]) -> dict[str, float]:
    """Concatenate cached per-clip events and compute bAcc per event type."""
    out = {}
    for ev in EVENT_NAMES:
        preds = []
        targets = []
        for c in clips:
            if ev in c["events"]:
                p, t = c["events"][ev]
                preds.append(p)
                targets.append(t)
        if not preds:
            continue
        preds_cat = torch.cat(preds)
        targets_cat = torch.cat(targets)
        out[ev] = _compute_bacc(preds_cat, targets_cat)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_pt", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_config", default="vap/conf/stereo_home_dev.yaml")
    parser.add_argument("--video_dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    if "session" not in df.columns:
        raise ValueError("CSV must have a 'session' column")
    sessions_per_clip = df["session"].astype(str).tolist()
    print(f"{len(sessions_per_clip)} clips, {len(set(sessions_per_clip))} unique sessions.")

    has_video = args.video_dim is not None and args.video_dim > 0
    module = _load_module(Path(args.ckpt), cfg, device)
    cached = _run_inference_and_cache_events(module, loader, has_video, sessions_per_clip)
    print(f"Cached events for {len(cached)} clips.")

    # Group clips by session
    by_session: dict[str, list[dict]] = defaultdict(list)
    for c in cached:
        by_session[c["session"]].append(c)
    sessions = list(by_session.keys())
    n_sess = len(sessions)
    print(f"Bootstrap over {n_sess} sessions, n_boot={args.n_boot}")

    # Point estimate
    point = _metrics_from_clips(cached)
    print("\n=== Point estimates (full test set) ===")
    print(f"{'Event':<6}  {'bAcc':>9}")
    for ev in EVENT_NAMES:
        if ev in point:
            print(f"{ev.upper():<6}  {point[ev]:>9.4f}")

    # Bootstrap
    boot_bacc: dict[str, list[float]] = defaultdict(list)
    for b in tqdm(range(args.n_boot), desc="Bootstrap"):
        sampled = [random.choice(sessions) for _ in range(n_sess)]
        clips = [c for s in sampled for c in by_session[s]]
        agg = _metrics_from_clips(clips)
        for ev in EVENT_NAMES:
            if ev in agg:
                boot_bacc[ev].append(agg[ev])

    # 95% CI per metric
    print("\n=== 95% CI from bootstrap ===")
    print(f"{'Event':<6}  {'bAcc mean':>10}  {'95% CI bAcc':>22}")
    for ev in EVENT_NAMES:
        if ev not in boot_bacc:
            continue
        b = np.array(boot_bacc[ev])
        cb = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        print(f"{ev.upper():<6}  {np.mean(b):>10.4f}  [{cb[0]:>7.4f}, {cb[1]:>7.4f}]")


if __name__ == "__main__":
    main()
