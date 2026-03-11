from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


def _load_checkpoint(module: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys in checkpoint load: {len(missing)}")
    if unexpected:
        print(f"[warn] Unexpected keys in checkpoint load: {len(unexpected)}")


def _text_series(df: pd.DataFrame) -> pd.Series:
    cols = []
    for name in ("dataset", "audio_path_a", "audio_path_b", "audio_path", "session"):
        if name in df.columns:
            cols.append(df[name].fillna("").astype(str))
    if not cols:
        raise ValueError(
            "Could not find expected columns: "
            "dataset/audio_path_a/audio_path_b/audio_path/session"
        )
    text = cols[0]
    for c in cols[1:]:
        text = text + " " + c
    return text.str.lower()


def _contains_any(s: pd.Series, keywords: tuple[str, ...]) -> pd.Series:
    mask = pd.Series(False, index=s.index)
    for k in keywords:
        mask |= s.str.contains(k.lower(), regex=False)
    return mask


def _split_test_csv(
    test_csv_path: Path,
    improvised_keywords: tuple[str, ...],
    naturalistic_keywords: tuple[str, ...],
) -> dict[str, Path]:
    df = pd.read_csv(test_csv_path)
    text = _text_series(df)

    improvised_mask = _contains_any(text, improvised_keywords)
    naturalistic_mask = _contains_any(text, naturalistic_keywords)

    improvised_df = df[improvised_mask].copy()
    naturalistic_df = df[naturalistic_mask].copy()
    overlap_count = int((improvised_mask & naturalistic_mask).sum())
    unmatched_count = int((~(improvised_mask | naturalistic_mask)).sum())

    improvised_path = test_csv_path.with_name(
        f"{test_csv_path.stem}_improvised{test_csv_path.suffix}"
    )
    naturalistic_path = test_csv_path.with_name(
        f"{test_csv_path.stem}_naturalistic{test_csv_path.suffix}"
    )
    improvised_df.to_csv(improvised_path, index=False)
    naturalistic_df.to_csv(naturalistic_path, index=False)

    print("\n=== Split Summary ===")
    print(f"input_rows: {len(df)}")
    print(f"improvised_rows: {len(improvised_df)} -> {improvised_path}")
    print(f"naturalistic_rows: {len(naturalistic_df)} -> {naturalistic_path}")
    print(f"unmatched_rows: {unmatched_count}")
    print(f"overlap_rows: {overlap_count}")

    return {
        "improvised": improvised_path,
        "naturalistic": naturalistic_path,
    }


def _resolve_use_visual(mode: str, model_video_dim: int) -> bool:
    mode_l = str(mode).strip().lower()
    if mode_l == "auto":
        return model_video_dim > 0
    if mode_l in {"true", "1", "yes", "on"}:
        return True
    if mode_l in {"false", "0", "no", "off"}:
        return False
    raise ValueError("use_visual must be one of: auto, true, false")


def _model_forward(
    module: torch.nn.Module,
    batch: dict[str, Any],
    use_visual: bool,
    model_video_dim: int,
):
    if model_video_dim <= 0:
        return module.model(batch["waveform"])

    if "video_features_a" not in batch or "video_features_b" not in batch:
        raise KeyError(
            "Batch does not contain video_features_a/video_features_b. "
            "Use a datamodule/dataset that provides visual features."
        )

    if use_visual:
        return module.model(
            batch["waveform"],
            video_features_a=batch["video_features_a"],
            video_features_b=batch["video_features_b"],
        )

    # Disable visual contribution while keeping visual model input shapes valid.
    return module.model(
        batch["waveform"],
        video_features_a=torch.zeros_like(batch["video_features_a"]),
        video_features_b=torch.zeros_like(batch["video_features_b"]),
    )


def _evaluate_single_split(
    module: torch.nn.Module,
    cfg: DictConfig,
    csv_path: Path,
    split_name: str,
    use_visual: bool,
    model_video_dim: int,
    batch_size: int,
    num_workers: int,
) -> None:
    cfg.datamodule.test_path = str(csv_path)
    cfg.datamodule.batch_size = int(batch_size)
    cfg.datamodule.num_workers = int(num_workers)

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    total_examples = 0
    vap_loss_sum = 0.0
    va_loss_sum = 0.0

    metric = getattr(module, "test_metric", None)
    if metric is not None:
        metric.reset()

    with torch.inference_mode():
        for batch in tqdm(test_loader, desc=f"Evaluating {split_name}"):
            batch = _to_device(batch, module.device)
            out = _model_forward(
                module=module,
                batch=batch,
                use_visual=use_visual,
                model_video_dim=model_video_dim,
            )

            labels = module.model.extract_labels(batch["vad"])
            vap_loss = module.model.objective.loss_vap(
                out["logits"], labels, reduction="mean"
            )
            va_loss = module.model.objective.loss_vad(out["vad"], batch["vad"])

            if metric is not None:
                probs = module.model.objective.get_probs(out["logits"])
                metric.update_batch(probs, batch["vad"])

            bsz = int(batch["waveform"].shape[0])
            total_examples += bsz
            vap_loss_sum += float(vap_loss) * bsz
            va_loss_sum += float(va_loss) * bsz

    if total_examples == 0:
        print(f"\n=== {split_name} ===")
        print("No rows in this split; skipping.")
        return

    print(f"\n=== {split_name} Losses ===")
    print(f"test_loss_vap:   {vap_loss_sum / total_examples:.6f}")
    print(f"test_loss_vad:   {va_loss_sum / total_examples:.6f}")
    print(f"test_loss_total: {(vap_loss_sum + va_loss_sum) / total_examples:.6f}")

    if metric is None:
        print("No test_metric configured; skipping accuracy metrics.")
        return

    scores = metric.compute()
    metric.reset()

    print(f"=== {split_name} Metrics ===")
    for event_name, score in scores.items():
        acc0 = float(score["acc"][0])
        acc1 = float(score["acc"][1])
        bacc = (acc0 + acc1) / 2.0
        f1 = float(score["f1"])
        print(
            f"{event_name}: acc0={acc0:.4f} acc1={acc1:.4f} "
            f"bacc={bacc:.4f} f1={f1:.4f}"
        )


@hydra.main(version_base=None, config_path="vap/conf", config_name="evaluate")
def main(cfg_eval: DictConfig) -> None:
    checkpoint_path = Path(to_absolute_path(str(cfg_eval.runtime.checkpoint_path)))
    test_csv_path = Path(to_absolute_path(str(cfg_eval.runtime.test_csv_path)))
    model_config_path = Path(to_absolute_path("vap/conf/stereo_home_dev.yaml"))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Config not found: {model_config_path}")

    cfg = OmegaConf.load(model_config_path)

    module = instantiate(cfg.module)
    if getattr(module, "test_metric", None) is None and "val_metric" in cfg.module:
        module.test_metric = instantiate(cfg.module.val_metric)

    _load_checkpoint(module, checkpoint_path)

    device_opt = str(cfg_eval.runtime.device).lower()
    if device_opt == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(cfg_eval.runtime.device))
    module = module.to(device)
    module.eval()

    model_video_dim = int(getattr(module.model, "video_dim", 0))
    use_visual = _resolve_use_visual(cfg_eval.runtime.use_visual, model_video_dim)
    effective_visual = use_visual and model_video_dim > 0

    print("\n=== Evaluation Setup ===")
    print(f"checkpoint: {checkpoint_path}")
    print("model_config: vap/conf/stereo_home_dev.yaml (hardcoded)")
    print(f"test_csv: {test_csv_path}")
    print(f"device: {device}")
    print(f"model.video_dim: {model_video_dim}")
    print(f"use_visual (requested): {cfg_eval.runtime.use_visual}")
    print(f"use_visual (effective): {effective_visual}")
    if not use_visual and model_video_dim > 0:
        print("[info] visual model detected; visuals disabled via zeroed video features.")

    if bool(cfg_eval.runtime.evaluate_full_test):
        _evaluate_single_split(
            module,
            cfg,
            test_csv_path,
            split_name="full_test",
            use_visual=use_visual,
            model_video_dim=model_video_dim,
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )

    if bool(cfg_eval.runtime.split_test_by_domain):
        improvised_keywords = tuple(cfg_eval.runtime.improvised_keywords)
        naturalistic_keywords = tuple(cfg_eval.runtime.naturalistic_keywords)
        split_paths = _split_test_csv(
            test_csv_path=test_csv_path,
            improvised_keywords=improvised_keywords,
            naturalistic_keywords=naturalistic_keywords,
        )
        _evaluate_single_split(
            module,
            cfg,
            split_paths["improvised"],
            "improvised",
            use_visual=use_visual,
            model_video_dim=model_video_dim,
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )
        _evaluate_single_split(
            module,
            cfg,
            split_paths["naturalistic"],
            "naturalistic",
            use_visual=use_visual,
            model_video_dim=model_video_dim,
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )


if __name__ == "__main__":
    main()
