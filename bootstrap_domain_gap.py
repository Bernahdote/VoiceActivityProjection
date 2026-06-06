"""
Stratified bootstrap CI on the improvised vs naturalistic gap for a single model.

For each iteration:
  - Resample improvised sessions with replacement, compute metric on imp clips
  - Resample naturalistic sessions with replacement, compute metric on nat clips
  - Store Δ = metric(improvised) - metric(naturalistic)
Report point estimate + 95% CI for the gap on HS/LS/SP bAcc + per-clip VAP loss.

Usage:
  uv run python bootstrap_domain_gap.py \\
      --ckpt /path/to/checkpoint.ckpt \\
      --test_pt /path/to/test_dir \\
      --test_csv /path/to/sliding.csv \\
      [--video_dim 364] [--n_boot 1000]
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


def _run_and_cache_events(module, loader, has_video, sessions, domains):
    metric = module.val_metric
    cached = []
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
            per_clip_loss = module.model.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).mean(dim=-1)
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
                    "session": sessions[clip_idx],
                    "domain": domains[clip_idx],
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
    if clips:
        out["loss"] = float(np.mean([c["loss"] for c in clips]))
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
    # Derive domain from audio_path_a substring
    def _domain(p: str) -> str:
        p = p.lower()
        if "improvised" in p:
            return "improvised"
        if "naturalistic" in p:
            return "naturalistic"
        return "other"
    domains_per_clip = [_domain(p) for p in df["audio_path_a"].astype(str).tolist()]

    n_imp_clips = sum(1 for d in domains_per_clip if d == "improvised")
    n_nat_clips = sum(1 for d in domains_per_clip if d == "naturalistic")
    print(f"Clips: total={len(sessions_per_clip)}, improvised={n_imp_clips}, naturalistic={n_nat_clips}")

    has_video = args.video_dim is not None and args.video_dim > 0
    module = _load_module(Path(args.ckpt), cfg, device)
    cached = _run_and_cache_events(module, loader, has_video, sessions_per_clip, domains_per_clip)

    # Group by session AND track which domain each session belongs to
    by_session_imp = defaultdict(list)
    by_session_nat = defaultdict(list)
    for c in cached:
        if c["domain"] == "improvised":
            by_session_imp[c["session"]].append(c)
        elif c["domain"] == "naturalistic":
            by_session_nat[c["session"]].append(c)

    sessions_imp = list(by_session_imp.keys())
    sessions_nat = list(by_session_nat.keys())
    print(f"Sessions: improvised={len(sessions_imp)}, naturalistic={len(sessions_nat)}")

    clips_imp = [c for c in cached if c["domain"] == "improvised"]
    clips_nat = [c for c in cached if c["domain"] == "naturalistic"]

    # Point estimates
    point_imp = _metrics_from_clips(clips_imp)
    point_nat = _metrics_from_clips(clips_nat)
    metric_keys = EVENT_NAMES + ["loss"]
    print("\n=== Point estimates ===")
    print(f"{'Metric':<6}  {'Imp':>10}  {'Nat':>10}  {'Δ (Imp-Nat)':>13}")
    for ev in metric_keys:
        if ev in point_imp and ev in point_nat:
            print(f"{ev.upper():<6}  {point_imp[ev]:>10.4f}  {point_nat[ev]:>10.4f}  {point_imp[ev]-point_nat[ev]:>+13.4f}")

    # Bootstrap: resample sessions within each domain independently
    deltas = defaultdict(list)
    n_imp_sess = len(sessions_imp)
    n_nat_sess = len(sessions_nat)
    for _ in tqdm(range(args.n_boot), desc="Bootstrap"):
        sampled_imp = [random.choice(sessions_imp) for _ in range(n_imp_sess)]
        sampled_nat = [random.choice(sessions_nat) for _ in range(n_nat_sess)]
        clips_imp_b = [c for s in sampled_imp for c in by_session_imp[s]]
        clips_nat_b = [c for s in sampled_nat for c in by_session_nat[s]]
        agg_imp = _metrics_from_clips(clips_imp_b)
        agg_nat = _metrics_from_clips(clips_nat_b)
        for ev in metric_keys:
            if ev in agg_imp and ev in agg_nat:
                deltas[ev].append(agg_imp[ev] - agg_nat[ev])

    print("\n=== Stratified bootstrap: 95% CI on Δ (Improvised - Naturalistic) ===")
    print(f"{'Metric':<6}  {'Δ mean':>10}  {'95% CI Δ':>22}")
    for ev in metric_keys:
        if ev not in deltas:
            continue
        d = np.array(deltas[ev])
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        print(f"{ev.upper():<6}  {np.mean(d):>+10.4f}  [{ci[0]:>+7.4f}, {ci[1]:>+7.4f}]")


if __name__ == "__main__":
    main()
