from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from vap.modules.lightning_module import VAPModule


SP_N_SEEDS = 10  # number of seeds for shift-prediction metric


def _split_test_csv(test_csv_path: Path) -> dict[str, Path]:
    df = pd.read_csv(test_csv_path)
    text = df.apply(
        lambda r: " ".join(str(r.get(c, "")) for c in ("dataset", "audio_path_a", "audio_path_b", "session") if c in df.columns),
        axis=1,
    ).str.lower()

    improvised_mask = text.str.contains("improvised", regex=False)
    naturalistic_mask = text.str.contains("naturalistic", regex=False)

    improvised_path = test_csv_path.with_name(f"{test_csv_path.stem}_improvised{test_csv_path.suffix}")
    naturalistic_path = test_csv_path.with_name(f"{test_csv_path.stem}_naturalistic{test_csv_path.suffix}")
    df[improvised_mask].to_csv(improvised_path, index=False)
    df[naturalistic_mask].to_csv(naturalistic_path, index=False)

    return {"improvised": improvised_path, "naturalistic": naturalistic_path}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _load_checkpoint(module: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys in checkpoint load: {len(missing)}")
        for k in missing:
            print(f"  missing: {k}")
    if unexpected:
        print(f"[warn] Unexpected keys in checkpoint load: {len(unexpected)}")
        for k in unexpected:
            print(f"  unexpected: {k}")


def _run_inference(
    module: torch.nn.Module,
    loader,
) -> tuple[list[dict], float, float, float]:
    """Run the model forward pass once and cache (probs, vad) per batch.

    Returns:
        cached   : list of {"probs": dict, "vad": Tensor} stored on CPU
        total_examples, vap_loss_sum, va_loss_sum
    """
    cached: list[dict] = []
    total_examples = 0
    vap_loss_sum = 0.0
    va_loss_sum = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = _to_device(batch, module.device)
            out = module.model(
                batch["waveform"],
                video_features_a=batch["video_features_a"],
                video_features_b=batch["video_features_b"],
            )

            labels = module.model.extract_labels(batch["vad"])
            vap_loss = module.model.objective.loss_vap(
                out["logits"], labels, reduction="mean"
            )
            va_loss = module.model.objective.loss_vad(out["vad"], batch["vad"])
            probs = module.model.objective.get_probs(out["logits"])

            bsz = int(batch["waveform"].shape[0])
            total_examples += bsz
            vap_loss_sum += float(vap_loss) * bsz
            va_loss_sum += float(va_loss) * bsz

            # Move to CPU so GPU memory is freed between batches
            cached.append({
                "probs": {k: v.cpu() for k, v in probs.items()},
                "vad": batch["vad"].cpu(),
            })

    return cached, total_examples, vap_loss_sum, va_loss_sum


def _compute_scores(metric, cached: list[dict], seed: int | None = None) -> dict:
    """Replay cached inference results through the metric.

    If seed is given, Python's random state is fixed before each replay so
    that the stochastic SP negative sampling is reproducible.
    """
    if seed is not None:
        random.seed(seed)
    metric.reset()
    for r in cached:
        metric.update_batch(r["probs"], r["vad"])
    scores = metric.compute()
    metric.reset()
    return scores


def _evaluate(
    module: torch.nn.Module,
    cfg: DictConfig,
    csv_path: Path,
    batch_size: int,
    num_workers: int,
) -> None:
    cfg.datamodule.test_path = str(csv_path)
    cfg.datamodule.batch_size = int(batch_size)
    cfg.datamodule.num_workers = int(num_workers)

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    metric = getattr(module, "val_metric", None)

    # ── inference (model forward pass — done once) ────────────────────────────
    cached, total_examples, vap_loss_sum, va_loss_sum = _run_inference(module, loader)

    print(f"\nloss_vap:   {vap_loss_sum / total_examples:.6f}")
    print(f"loss_vad:   {va_loss_sum / total_examples:.6f}")
    print(f"loss_total: {(vap_loss_sum + va_loss_sum) / total_examples:.6f}")

    if metric is None:
        print("No metric configured.")
        return

    # ── deterministic metrics: HS and LS ─────────────────────────────────────
    # equal_hold_shift=False → all holds used, no random sampling → one run suffices
    scores_det = _compute_scores(metric, cached, seed=None)

    LABELS = {
        "hs": ("Hold",      "Shift"),
        "ls": ("Short",     "Long"),
        "sp": ("Pre-hold",  "Pre-shift"),
        "bp": ("Non-BC",    "Backchannel"),
    }

    # Rerun to get raw sample counts for sanity checking
    metric.reset()
    for r in cached:
        metric.update_batch(r["probs"], r["vad"])
    _, targets_flat = metric._flatten()
    for event_name in ("hs", "ls"):
        if event_name in targets_flat:
            t = targets_flat[event_name]
            n0 = int((t == 0).sum())
            n1 = int((t == 1).sum())
            print(f"  [{event_name}] n_class0={n0}  n_class1={n1}  total={n0+n1}")
    metric.reset()

    print()
    for event_name in ("hs", "ls"):
        score = scores_det[event_name]
        acc0 = float(score["acc"][0])
        acc1 = float(score["acc"][1])
        bacc = (acc0 + acc1) / 2.0
        f1   = float(score["f1"])
        lbl0, lbl1 = LABELS[event_name]
        print(f"{event_name.upper()}:  {lbl0}={acc0:.4f}  {lbl1}={acc1:.4f}  bAcc={bacc:.4f}  F1={f1:.4f}")

    # ── SP and BP: stochastic negative sampling → run with seeds 1‥SP_N_SEEDS ─
    sp_baccs, sp_f1s, sp_acc0s, sp_acc1s = [], [], [], []
    bp_baccs, bp_f1s, bp_acc0s, bp_acc1s = [], [], [], []
    for seed in range(1, SP_N_SEEDS + 1):
        scores_seed = _compute_scores(metric, cached, seed=seed)

        s = scores_seed["sp"]
        sp_acc0s.append(float(s["acc"][0]))
        sp_acc1s.append(float(s["acc"][1]))
        sp_baccs.append((sp_acc0s[-1] + sp_acc1s[-1]) / 2.0)
        sp_f1s.append(float(s["f1"]))

        if "bp" in scores_seed:
            b = scores_seed["bp"]
            bp_acc0s.append(float(b["acc"][0]))
            bp_acc1s.append(float(b["acc"][1]))
            bp_baccs.append((bp_acc0s[-1] + bp_acc1s[-1]) / 2.0)
            bp_f1s.append(float(b["f1"]))

    lbl0, lbl1 = LABELS["sp"]
    print(
        f"SP:  {lbl0}={np.mean(sp_acc0s):.4f}±{np.std(sp_acc0s):.4f}"
        f"  {lbl1}={np.mean(sp_acc1s):.4f}±{np.std(sp_acc1s):.4f}"
        f"  bAcc={np.mean(sp_baccs):.4f}±{np.std(sp_baccs):.4f}"
        f"  F1={np.mean(sp_f1s):.4f}±{np.std(sp_f1s):.4f}"
        f"  (n={SP_N_SEEDS} seeds)"
    )

    if bp_baccs:
        lbl0, lbl1 = LABELS["bp"]
        print(
            f"BP:  {lbl0}={np.mean(bp_acc0s):.4f}±{np.std(bp_acc0s):.4f}"
            f"  {lbl1}={np.mean(bp_acc1s):.4f}±{np.std(bp_acc1s):.4f}"
            f"  bAcc={np.mean(bp_baccs):.4f}±{np.std(bp_baccs):.4f}"
            f"  F1={np.mean(bp_f1s):.4f}±{np.std(bp_f1s):.4f}"
            f"  (n={SP_N_SEEDS} seeds)"
        )


@hydra.main(version_base=None, config_path="vap/conf", config_name="evaluate")
def main(cfg_eval: DictConfig) -> None:
    checkpoint_path = Path(to_absolute_path(str(cfg_eval.runtime.checkpoint_path)))
    test_csv_path = Path(to_absolute_path(str(cfg_eval.runtime.test_csv_path))) if cfg_eval.runtime.test_csv_path is not None else None
    test_pt_path = cfg_eval.runtime.get("test_pt_path", None)
    if test_pt_path is not None:
        test_pt_path = Path(to_absolute_path(str(test_pt_path)))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if test_pt_path is None and not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    cfg = cfg_eval
    module = VAPModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    if cfg_eval.runtime.base:
        module.model.feature_projection = torch.nn.Identity()
    module.val_metric = instantiate(cfg.module.val_metric)

    device_opt = str(cfg_eval.runtime.device).lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_opt == "auto" else torch.device(device_opt)
    module = module.to(device)
    module.eval()

    print(f"checkpoint: {checkpoint_path}")
    print(f"device:     {device}")

    kwargs = dict(module=module, cfg=cfg, batch_size=int(cfg_eval.runtime.batch_size), num_workers=int(cfg_eval.runtime.num_workers))

    if test_pt_path is not None:
        # Use .pt directory directly — no improvised/naturalistic split
        print(f"test_pt:    {test_pt_path}")
        print("\n=== Full ===")
        _evaluate(csv_path=test_pt_path, **kwargs)
    else:
        print(f"test_csv:   {test_csv_path}")
        print("\n=== Full ===")
        _evaluate(csv_path=test_csv_path, **kwargs)

        split_paths = _split_test_csv(test_csv_path)
        print("\n=== Improvised ===")
        _evaluate(csv_path=split_paths["improvised"], **kwargs)
        print("\n=== Naturalistic ===")
        _evaluate(csv_path=split_paths["naturalistic"], **kwargs)


if __name__ == "__main__":
    main()
