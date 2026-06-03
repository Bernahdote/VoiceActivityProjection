"""
Paired bootstrap 95% CI on the delta between two model checkpoints.

For each bootstrap iteration:
  - Sample N sessions with replacement
  - Compute bAcc for BOTH models on the same sampled sessions
  - Store delta = bAcc(M1) - bAcc(M0)

This is paired — both models see the same sessions per iteration — so the
shared session variance cancels out. The CI on the delta is tighter than
comparing two independent CIs.

Usage:
  uv run python bootstrap_paired.py \\
      --ckpt0 /path/to/baseline.ckpt \\
      --ckpt1 /path/to/improved.ckpt \\
      --test_pt0 /path/to/test_pt0 \\
      --test_pt1 /path/to/test_pt1 \\
      --test_csv /path/to/sliding.csv \\
      [--video_dim0 N --video_dim1 M] [--n_boot 1000]
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


def _run_and_cache_events(module, loader, has_video: bool, sessions: list[str]):
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
            labels = module.model.extract_labels(batch["vad"])
            # Per-clip VAP loss (no reduction so we can keep per-clip)
            per_clip_loss = module.model.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).mean(dim=-1)  # (B,)
            bsz = batch["waveform"].shape[0]
            for b in range(bsz):
                metric.reset()
                probs_clip = {k: v[b:b+1] for k, v in probs.items()}
                vad_clip = batch["vad"][b:b+1]
                metric.update_batch(probs_clip, vad_clip)
                clip_events = {}
                for ev in EVENT_NAMES:
                    if metric.preds[ev]:
                        p = torch.cat(metric.preds[ev]).detach().cpu()
                        t = torch.cat(metric.targets[ev]).detach().cpu()
                        clip_events[ev] = (p, t)
                cached.append({
                    "session": sessions[clip_idx] if clip_idx < len(sessions) else str(clip_idx),
                    "events": clip_events,
                    "loss": float(per_clip_loss[b]),
                })
                clip_idx += 1
    metric.reset()
    return cached


def _compute_bacc(preds, targets, threshold=0.5):
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


def _metrics_from_clips(clips):
    out = {}
    for ev in EVENT_NAMES:
        preds, targets = [], []
        for c in clips:
            if ev in c["events"]:
                p, t = c["events"][ev]
                preds.append(p)
                targets.append(t)
        if not preds:
            continue
        out[ev] = _compute_bacc(torch.cat(preds), torch.cat(targets))
    # Per-clip loss averaged across all clips in the sample
    out["loss"] = float(np.mean([c["loss"] for c in clips]))
    return out


def _make_loader(test_pt: str, cfg, batch_size: int, num_workers: int):
    cfg = cfg.copy()
    cfg.datamodule.test_path = test_pt
    cfg.datamodule.batch_size = batch_size
    cfg.datamodule.num_workers = num_workers
    dm = instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup("test")
    return dm.test_dataloader()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt0", required=True, help="Baseline model checkpoint")
    parser.add_argument("--ckpt1", required=True, help="Improved model checkpoint")
    parser.add_argument("--test_pt0", required=True, help=".pt directory for baseline (e.g. test_3_all_200 for audio)")
    parser.add_argument("--test_pt1", required=True, help=".pt directory for improved model (matches its training)")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_config", default="vap/conf/stereo_home_dev.yaml")
    parser.add_argument("--video_dim0", type=int, default=None, help="video_dim for ckpt0 (omit for audio-only)")
    parser.add_argument("--video_dim1", type=int, default=None, help="video_dim for ckpt1 (omit for audio-only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd
    df = pd.read_csv(args.test_csv)
    if "session" not in df.columns:
        raise ValueError("CSV must have a 'session' column")
    sessions_per_clip = df["session"].astype(str).tolist()
    print(f"{len(sessions_per_clip)} clips, {len(set(sessions_per_clip))} unique sessions.")

    # Model 0
    cfg0 = OmegaConf.load(args.model_config)
    if args.video_dim0 is not None:
        cfg0.module.model.video_dim = args.video_dim0
    loader0 = _make_loader(args.test_pt0, cfg0, args.batch_size, args.num_workers)
    m0 = _load_module(Path(args.ckpt0), cfg0, device)
    print("Inference for model 0...")
    cached0 = _run_and_cache_events(m0, loader0, args.video_dim0 is not None and args.video_dim0 > 0, sessions_per_clip)
    del m0

    # Model 1
    cfg1 = OmegaConf.load(args.model_config)
    if args.video_dim1 is not None:
        cfg1.module.model.video_dim = args.video_dim1
    loader1 = _make_loader(args.test_pt1, cfg1, args.batch_size, args.num_workers)
    m1 = _load_module(Path(args.ckpt1), cfg1, device)
    print("Inference for model 1...")
    cached1 = _run_and_cache_events(m1, loader1, args.video_dim1 is not None and args.video_dim1 > 0, sessions_per_clip)
    del m1

    assert len(cached0) == len(cached1), "mismatched clip counts"

    by_sess0: dict[str, list[dict]] = defaultdict(list)
    by_sess1: dict[str, list[dict]] = defaultdict(list)
    for c0, c1 in zip(cached0, cached1):
        by_sess0[c0["session"]].append(c0)
        by_sess1[c1["session"]].append(c1)
    sessions = list(by_sess0.keys())
    n_sess = len(sessions)
    print(f"Bootstrap over {n_sess} sessions, n_boot={args.n_boot}")

    # Point deltas
    point0 = _metrics_from_clips(cached0)
    point1 = _metrics_from_clips(cached1)
    print("\n=== Point estimates ===")
    metric_keys = EVENT_NAMES + ["loss"]
    print(f"{'Metric':<6}  {'M0':>9}  {'M1':>9}  {'Δ (M1-M0)':>12}")
    for ev in metric_keys:
        if ev in point0 and ev in point1:
            print(f"{ev.upper():<6}  {point0[ev]:>9.4f}  {point1[ev]:>9.4f}  {point1[ev]-point0[ev]:>+12.4f}")

    # Bootstrap deltas
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in tqdm(range(args.n_boot), desc="Bootstrap"):
        sampled = [random.choice(sessions) for _ in range(n_sess)]
        clips0 = [c for s in sampled for c in by_sess0[s]]
        clips1 = [c for s in sampled for c in by_sess1[s]]
        agg0 = _metrics_from_clips(clips0)
        agg1 = _metrics_from_clips(clips1)
        for ev in metric_keys:
            if ev in agg0 and ev in agg1:
                deltas[ev].append(agg1[ev] - agg0[ev])

    # 95% CI on deltas
    print("\n=== Paired bootstrap: 95% CI on Δ (M1 - M0) ===")
    print(f"{'Metric':<6}  {'Δ mean':>10}  {'95% CI Δ':>22}  Significant?")
    for ev in metric_keys:
        if ev not in deltas:
            continue
        d = np.array(deltas[ev])
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        sig = "yes (excludes 0)" if (ci[0] > 0 or ci[1] < 0) else "no (includes 0)"
        print(f"{ev.upper():<6}  {np.mean(d):>+10.4f}  [{ci[0]:>+7.4f}, {ci[1]:>+7.4f}]  {sig}")


if __name__ == "__main__":
    main()
