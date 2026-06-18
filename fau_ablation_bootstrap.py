"""
Bootstrap 95% CI for FAU ablation importance.

For each of the 24 FAU dimensions, zero out both the FAU value (position i) and
its delta (position i + N_FAU), evaluate per-clip losses, then bootstrap by
session to get a 95% CI on the loss increase relative to the baseline (no
ablation).

Approach:
  1. Run inference 25 times (baseline + one per FAU dim), caching per-clip loss
     under each condition.
  2. Group clips by session.
  3. Bootstrap N=1000 iterations: sample sessions with replacement, compute
     mean(loss_ablated) - mean(loss_baseline) on the sample for each FAU.
  4. Report point estimate + 95% CI per FAU.

Usage:
  uv run python fau_ablation_bootstrap.py
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

FAU_NAMES = [
    "InnerBrowRaiser (AU1)",
    "OuterBrowRaiser (AU2)",
    "BrowLowerer (AU4)",
    "UpperLidRaiser (AU5)",
    "CheekRaiser (AU6)",
    "LidTightener (AU7)",
    "NoseWrinkler (AU9)",
    "UpperLipRaiser (AU10)",
    "LipCornerPull (AU12)",
    "CheekPuffer (AU13)",
    "Dimpler (AU14)",
    "LipCornerDepressor (AU15)",
    "LowerLipDepressor (AU16)",
    "ChinRaiser (AU17)",
    "LipPuckerer (AU18)",
    "LipStretcher (AU20)",
    "LipFunneler (AU22)",
    "LipTightener (AU23)",
    "LipPressor (AU24)",
    "LipsParts (AU25)",
    "JawDrop (AU26)",
    "LipSuck (AU28)",
    "JawSideways (AU30)",
    "EyesClosed (AU43)",
]


# FAU-only model and its matching .pt test directory
CHECKPOINT = "/mnt/sdb/willem/VoiceActivityProjection/runs_new/VAP_debug/lc2fbkxc/checkpoints/epoch=6-step=22505.ckpt"
TEST_PT = "/mnt/sdb/willem/datasets/preprocessed/test_3_fauv_200"
TEST_CSV = "/mnt/sdb/willem/datasets/splits_200/test_sliding.csv"
N_FAU = 24      # FAU value dimensions; delta-3 doubles the layout
N_BOOT = 1000   # bootstrap iterations
SEED = 42


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _load_checkpoint(module: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys: {len(unexpected)}")


def _per_clip_losses(module, loader, device, zeroed_dims: list[int] | None, desc: str) -> list[float]:
    """Run inference and return per-clip losses (one value per clip in dataloader order)."""
    losses: list[float] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = _to_device(batch, device)
            if zeroed_dims is not None:
                batch["video_features_a"] = batch["video_features_a"].clone()
                batch["video_features_b"] = batch["video_features_b"].clone()
                for d in zeroed_dims:
                    batch["video_features_a"][..., d] = 0.0
                    batch["video_features_b"][..., d] = 0.0
            out = module.model(
                batch["waveform"],
                video_features_a=batch["video_features_a"],
                video_features_b=batch["video_features_b"],
            )
            labels = module.model.extract_labels(batch["vad"])
            # reduction="none" then mean over frames → per-clip loss
            per_clip = module.model.objective.loss_vap(
                out["logits"], labels, reduction="none"
            ).mean(dim=-1)  # (B,)
            losses.extend(per_clip.detach().cpu().tolist())
    return losses


@hydra.main(version_base=None, config_path="vap/conf", config_name="evaluate")
def main(cfg_eval: DictConfig) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    checkpoint_path = Path(to_absolute_path(CHECKPOINT))
    test_pt_path = Path(to_absolute_path(TEST_PT))
    test_csv_path = Path(to_absolute_path(TEST_CSV))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_pt_path.is_dir():
        raise NotADirectoryError(f"Test .pt directory not found: {test_pt_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    cfg_eval.module.model.video_dim = N_FAU * 2  # 48 with delta3
    module = instantiate(cfg_eval.module)
    _load_checkpoint(module, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    module.eval()

    cfg_eval.datamodule.test_path = str(test_pt_path)
    cfg_eval.datamodule.batch_size = int(cfg_eval.runtime.batch_size)
    cfg_eval.datamodule.num_workers = int(cfg_eval.runtime.num_workers)
    datamodule = instantiate(cfg_eval.datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    # Load session ids in same order as clips (CSV row order matches .pt index)
    import pandas as pd
    df = pd.read_csv(test_csv_path)
    sessions_per_clip = df["session"].astype(str).tolist()

    print(f"checkpoint: {checkpoint_path}")
    print(f"test_pt:    {test_pt_path}")
    print(f"device:     {device}\n")

    # ── baseline (no ablation) ────────────────────────────────────────────
    baseline = _per_clip_losses(module, loader, device, zeroed_dims=None, desc="baseline")
    baseline = np.array(baseline)
    n_clips = len(baseline)
    if n_clips != len(sessions_per_clip):
        print(f"[warn] clip count {n_clips} != csv rows {len(sessions_per_clip)}")
    print(f"Baseline mean loss: {baseline.mean():.6f}\n")

    # ── each FAU ablation ────────────────────────────────────────────────
    ablated: dict[int, np.ndarray] = {}
    for i in range(N_FAU):
        losses = _per_clip_losses(
            module, loader, device, zeroed_dims=[i, i + N_FAU],
            desc=f"FAU {i:2d} {FAU_NAMES[i]}"
        )
        ablated[i] = np.array(losses)

    # ── group clip indices by session ────────────────────────────────────
    sess_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, s in enumerate(sessions_per_clip[:n_clips]):
        sess_to_indices[s].append(idx)
    sessions = list(sess_to_indices.keys())
    n_sess = len(sessions)
    print(f"\nBootstrap over {n_sess} sessions, N={N_BOOT}\n")

    # ── point estimate ────────────────────────────────────────────────────
    point_deltas = {i: float((ablated[i] - baseline).mean()) for i in range(N_FAU)}

    # ── bootstrap ─────────────────────────────────────────────────────────
    boot_deltas: dict[int, list[float]] = defaultdict(list)
    for _ in tqdm(range(N_BOOT), desc="Bootstrap"):
        sampled = [random.choice(sessions) for _ in range(n_sess)]
        idxs = np.array([k for s in sampled for k in sess_to_indices[s]])
        baseline_mean = baseline[idxs].mean()
        for i in range(N_FAU):
            boot_deltas[i].append(float(ablated[i][idxs].mean() - baseline_mean))

    # ── results ───────────────────────────────────────────────────────────
    print("\n=== FAU ablation: 95% CI on Δloss (ablated − baseline) ===")
    print(f"{'Idx':<4} {'AU':<28} {'Δ point':>10}  {'Δ mean':>10}  {'95% CI':>22}")
    rows = []
    for i in range(N_FAU):
        d = np.array(boot_deltas[i])
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        rows.append((i, FAU_NAMES[i], point_deltas[i], float(d.mean()), ci))

    for i, name, p, m, ci in rows:
        print(f"[{i:2d}] {name:<28} {p:>+10.4f}  {m:>+10.4f}  [{ci[0]:>+7.4f}, {ci[1]:>+7.4f}]")

    print("\n--- Ranked by importance (largest Δ first) ---")
    for i, name, p, m, ci in sorted(rows, key=lambda x: -x[3]):
        print(f"[{i:2d}] {name:<28} {p:>+10.4f}  {m:>+10.4f}  [{ci[0]:>+7.4f}, {ci[1]:>+7.4f}]")

    # ── plot ──────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    rows_sorted = sorted(rows, key=lambda x: -x[3])
    names = [r[1] for r in rows_sorted]
    means = [r[3] for r in rows_sorted]
    ci_lo = [r[4][0] for r in rows_sorted]
    ci_hi = [r[4][1] for r in rows_sorted]
    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    colors = ["tomato" if m > 0 else "steelblue" for m in means]

    plt.figure(figsize=(10, 8))
    y = np.arange(len(names))
    plt.barh(y, means[::-1], color=colors[::-1])
    plt.errorbar(means[::-1], y, xerr=[err_lo[::-1], err_hi[::-1]],
                 fmt="none", ecolor="black", capsize=3, linewidth=0.8)
    plt.yticks(y, names[::-1])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Δ VAP loss vs baseline (95% CI from N=1000 bootstrap over sessions)")
    plt.title("FAU ablation — importance by loss increase")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
