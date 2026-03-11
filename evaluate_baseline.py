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


def _split_test_csv(test_csv_path: Path) -> dict[str, Path]:
    df = pd.read_csv(test_csv_path)
    text = _text_series(df)

    improvised_mask = _contains_any(text, ("improvised",))
    naturalistic_mask = _contains_any(text, ("naturalistic",))

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
    return {"improvised": improvised_path, "naturalistic": naturalistic_path}


def _evaluate_single_split(
    module: torch.nn.Module,
    cfg: DictConfig,
    csv_path: Path,
    split_name: str,
    split_kind: str,
    batch_size: int,
    num_workers: int,
) -> None:
    if split_kind == "test":
        cfg.datamodule.test_path = str(csv_path)
    elif split_kind == "val":
        cfg.datamodule.test_path = str(csv_path)
    else:
        raise ValueError("split_kind must be one of: test, val")

    cfg.datamodule.batch_size = int(batch_size)
    cfg.datamodule.num_workers = int(num_workers)

    datamodule = instantiate(cfg.datamodule)
    datamodule.prepare_data()
    if split_kind == "test":
        datamodule.setup("test")
        loader = datamodule.test_dataloader()
        metric = getattr(module, "test_metric", None)
    else:
        datamodule.setup("test")
        loader = datamodule.test_dataloader()
        metric = getattr(module, "val_metric", None)

    total_examples = 0
    vap_loss_sum = 0.0
    va_loss_sum = 0.0

    if metric is not None:
        metric.reset()

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluating {split_name}"):
            batch = _to_device(batch, module.device)
            out = module.model(batch["waveform"])

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
    print(f"loss_vap:   {vap_loss_sum / total_examples:.6f}")
    print(f"loss_vad:   {va_loss_sum / total_examples:.6f}")
    print(f"loss_total: {(vap_loss_sum + va_loss_sum) / total_examples:.6f}")

    if metric is None:
        print("No metric configured; skipping accuracy metrics.")
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
    model_cfg = cfg_eval.runtime.get("model_config_path", "vap/conf/default_config.yaml")
    model_config_path = Path(to_absolute_path(str(model_cfg)))

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

    print("\n=== Evaluation Setup (Baseline) ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"model_config: {model_config_path}")
    print(f"test_csv: {test_csv_path}")
    print(f"device: {device}")

    val_csv_raw = cfg_eval.runtime.get("val_csv_path", None)
    if val_csv_raw is not None and str(val_csv_raw).strip() != "":
        val_csv_path = Path(to_absolute_path(str(val_csv_raw)))
        if not val_csv_path.is_file():
            raise FileNotFoundError(f"Validation CSV not found: {val_csv_path}")
        _evaluate_single_split(
            module=module,
            cfg=cfg,
            csv_path=val_csv_path,
            split_name="full_val",
            split_kind="val",
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )

    _evaluate_single_split(
        module=module,
        cfg=cfg,
        csv_path=test_csv_path,
        split_name="full_test",
        split_kind="test",
        batch_size=int(cfg_eval.runtime.batch_size),
        num_workers=int(cfg_eval.runtime.num_workers),
    )

    if bool(cfg_eval.runtime.split_test_by_domain):
        split_paths = _split_test_csv(test_csv_path=test_csv_path)
        _evaluate_single_split(
            module=module,
            cfg=cfg,
            csv_path=split_paths["improvised"],
            split_name="improvised",
            split_kind="test",
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )
        _evaluate_single_split(
            module=module,
            cfg=cfg,
            csv_path=split_paths["naturalistic"],
            split_name="naturalistic",
            split_kind="test",
            batch_size=int(cfg_eval.runtime.batch_size),
            num_workers=int(cfg_eval.runtime.num_workers),
        )


if __name__ == "__main__":
    main()
