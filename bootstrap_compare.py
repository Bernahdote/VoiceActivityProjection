"""
Bootstrap 95% confidence intervals for a single model's metrics.

For each bootstrap iteration:
  - Sample N sessions (conversations) with replacement
  - Compute HS/LS/SP/BP bAcc + F1 on the resampled clips
  - Store the metric values

Output:
  - Point estimate + 95% CI per metric

Usage:
  uv run python bootstrap_compare.py \\
      --ckpt /path/to/checkpoint.ckpt \\
      --test_pt /path/to/test_dir \\
      --test_csv /path/to/sliding.csv \\
      [--video_dim 306] [--n_boot 1000] [--sp_seeds 10]
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


SP_N_SEEDS_DEFAULT = 10


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _load_module(ckpt_path: Path, cfg, device):
    module = VAPModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    module.val_metric = instantiate(cfg.module.val_metric)
    module = module.to(device)
    module.eval()
    return module


def _run_inference(module, loader, has_video: bool, sessions: list[str]) -> list[dict]:
    cached: list[dict] = []
    clip_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
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
                cached.append({
                    "session": sessions[clip_idx] if clip_idx < len(sessions) else str(clip_idx),
                    "probs": {k: v[b:b+1].cpu() for k, v in probs.items()},
                    "vad": batch["vad"][b:b+1].cpu(),
                })
                clip_idx += 1
    return cached


def _metric_on_clips(metric, clips: list[dict], seed: int | None = None) -> dict:
    if seed is not None:
        random.seed(seed)
    metric.reset()
    for c in clips:
        metric.update_batch(c["probs"], c["vad"])
    scores = metric.compute()
    metric.reset()
    return scores


def _aggregate(metric, clips: list[dict], sp_seeds: int) -> dict[str, dict[str, float]]:
    det = _metric_on_clips(metric, clips, seed=None)
    out: dict[str, dict[str, float]] = {}
    for event in ("hs", "ls"):
        if event in det:
            acc = det[event]["acc"]
            out[event] = {"bAcc": float((acc[0] + acc[1]) / 2.0), "F1": float(det[event]["f1"])}
    sp_b, sp_f, bp_b, bp_f = [], [], [], []
    for s in range(1, sp_seeds + 1):
        sc = _metric_on_clips(metric, clips, seed=s)
        if "sp" in sc:
            a = sc["sp"]["acc"]; sp_b.append(float((a[0] + a[1]) / 2.0)); sp_f.append(float(sc["sp"]["f1"]))
        if "bp" in sc:
            a = sc["bp"]["acc"]; bp_b.append(float((a[0] + a[1]) / 2.0)); bp_f.append(float(sc["bp"]["f1"]))
    if sp_b:
        out["sp"] = {"bAcc": float(np.mean(sp_b)), "F1": float(np.mean(sp_f))}
    if bp_b:
        out["bp"] = {"bAcc": float(np.mean(bp_b)), "F1": float(np.mean(bp_f))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--test_pt", required=True, help="Test .pt directory")
    parser.add_argument("--test_csv", required=True, help="Test CSV (sliding) for session info")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--sp_seeds", type=int, default=SP_N_SEEDS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_config", default="vap/conf/stereo_home_dev.yaml")
    parser.add_argument("--video_dim", type=int, default=None,
                        help="Set for video models, omit for audio-only baseline.")
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
    cached = _run_inference(module, loader, has_video, sessions_per_clip)
    metric = getattr(module, "val_metric")

    by_session: dict[str, list[dict]] = defaultdict(list)
    for c in cached:
        by_session[c["session"]].append(c)
    sessions = list(by_session.keys())
    n_sess = len(sessions)
    print(f"Bootstrap over {n_sess} sessions, n_boot={args.n_boot}")

    # Point estimate
    point = _aggregate(metric, cached, args.sp_seeds)
    print("\n=== Point estimates (full test set) ===")
    print(f"{'Event':<6}  {'bAcc':>9}  {'F1':>9}")
    for ev in ("hs", "ls", "sp", "bp"):
        if ev in point:
            print(f"{ev.upper():<6}  {point[ev]['bAcc']:>9.4f}  {point[ev]['F1']:>9.4f}")

    # Bootstrap
    boot_bacc: dict[str, list[float]] = defaultdict(list)
    boot_f1: dict[str, list[float]] = defaultdict(list)
    for b in tqdm(range(args.n_boot), desc="Bootstrap"):
        sampled = [random.choice(sessions) for _ in range(n_sess)]
        clips = [c for s in sampled for c in by_session[s]]
        agg = _aggregate(metric, clips, args.sp_seeds)
        for ev in ("hs", "ls", "sp", "bp"):
            if ev in agg:
                boot_bacc[ev].append(agg[ev]["bAcc"])
                boot_f1[ev].append(agg[ev]["F1"])

    # 95% CI per metric
    print("\n=== 95% CI from bootstrap ===")
    print(f"{'Event':<6}  {'bAcc mean':>10}  {'95% CI bAcc':>22}  {'F1 mean':>10}  {'95% CI F1':>22}")
    for ev in ("hs", "ls", "sp", "bp"):
        if ev not in boot_bacc:
            continue
        b = np.array(boot_bacc[ev])
        f = np.array(boot_f1[ev])
        cb = (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))
        cf = (float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5)))
        print(f"{ev.upper():<6}  {np.mean(b):>10.4f}  [{cb[0]:>7.4f}, {cb[1]:>7.4f}]  "
              f"{np.mean(f):>10.4f}  [{cf[0]:>7.4f}, {cf[1]:>7.4f}]")


if __name__ == "__main__":
    main()
