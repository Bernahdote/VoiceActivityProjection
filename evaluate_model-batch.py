from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from vap.modules.lightning_module import VAPModule


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
    if metric is not None:
        metric.reset()

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

            if metric is not None:
                probs = module.model.objective.get_probs(out["logits"])
                metric.update_batch(probs, batch["vad"])

            bsz = int(batch["waveform"].shape[0])
            total_examples += bsz
            vap_loss_sum += float(vap_loss) * bsz
            va_loss_sum += float(va_loss) * bsz

    print(f"\nloss_vap:   {vap_loss_sum / total_examples:.6f}")
    print(f"loss_vad:   {va_loss_sum / total_examples:.6f}")
    print(f"loss_total: {(vap_loss_sum + va_loss_sum) / total_examples:.6f}")

    if metric is None:
        print("No metric configured.")
        return

    for event_name in metric.EVENT_NAMES:
        n = sum(len(t) for t in metric.preds[event_name])
        print(f"  {event_name}: {n} samples")

    scores = metric.compute()
    metric.reset()

    LABELS = {
        "hs": ("Hold",      "Shift"),
        "ls": ("Long",      "Short"),
        "sp": ("Pre-hold",  "Pre-shift"),
        "bp": ("Non-BC",    "Backchannel"),
    }

    print()
    for event_name, score in scores.items():
        acc0 = float(score["acc"][0])
        acc1 = float(score["acc"][1])
        bacc = (acc0 + acc1) / 2.0
        f1   = float(score["f1"])
        lbl0, lbl1 = LABELS.get(event_name, ("acc0", "acc1"))
        print(f"{event_name.upper()}:  {lbl0}={acc0:.4f}  {lbl1}={acc1:.4f}  bAcc={bacc:.4f}  F1={f1:.4f}")


@hydra.main(version_base=None, config_path="vap/conf", config_name="evaluate")
def main(cfg_eval: DictConfig) -> None:
    checkpoint_path = Path(to_absolute_path(str(cfg_eval.runtime.checkpoint_path)))
    test_csv_path = Path(to_absolute_path(str(cfg_eval.runtime.test_csv_path)))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    cfg = cfg_eval
    module = VAPModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    if cfg_eval.runtime.base:
        # newMaster (audio-only) checkpoints used nn.Identity() for feature_projection.
        # Current branch replaced it with an MLP whose weights are absent from the checkpoint
        # and would be randomly initialized. Reset to Identity to match training.
        module.model.feature_projection = torch.nn.Identity()
    if "val_metric" in cfg.module:
        module.val_metric = instantiate(cfg.module.val_metric)

    device_opt = str(cfg_eval.runtime.device).lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_opt == "auto" else torch.device(device_opt)
    module = module.to(device)
    module.eval()

    print(f"checkpoint: {checkpoint_path}")
    print(f"test_csv:   {test_csv_path}")
    print(f"device:     {device}")

    kwargs = dict(module=module, cfg=cfg, batch_size=int(cfg_eval.runtime.batch_size), num_workers=int(cfg_eval.runtime.num_workers))

    print("\n=== Full ===")
    _evaluate(csv_path=test_csv_path, **kwargs)

    split_paths = _split_test_csv(test_csv_path)
    print("\n=== Improvised ===")
    _evaluate(csv_path=split_paths["improvised"], **kwargs)
    print("\n=== Naturalistic ===")
    _evaluate(csv_path=split_paths["naturalistic"], **kwargs)


if __name__ == "__main__":
    main()
