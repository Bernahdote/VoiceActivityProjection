import logging
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


log: logging.Logger = logging.getLogger(__name__)


def _load_state_dict(module: torch.nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(
            f"Unsupported checkpoint format in {checkpoint_path}. Expected a dict."
        )

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys when loading checkpoint: %d", len(missing))
    if unexpected:
        log.warning("Unexpected keys when loading checkpoint: %d", len(unexpected))


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


@torch.inference_mode()
def _validation_loss_with_zero_visuals(
    module: torch.nn.Module, val_loader: Any, device: torch.device
) -> tuple[float, dict[str, float]]:
    module = module.to(device)
    module.eval()

    total_examples = 0
    total_loss_sum = 0.0

    metric = getattr(module, "val_metric", None)
    if metric:
        metric.reset()
        if hasattr(metric, "to"):
            metric = metric.to(device)
            module.val_metric = metric

    for batch in tqdm(val_loader, desc="Validation", leave=False):
        batch = _to_device(batch, device)
        batch["video_features_a"] = torch.zeros_like(batch["video_features_a"])
        batch["video_features_b"] = torch.zeros_like(batch["video_features_b"])

        out = module.model(
            batch["waveform"],
            video_features_a=batch["video_features_a"],
            video_features_b=batch["video_features_b"],
        )
        labels = module.model.extract_labels(batch["vad"])
        vap_loss = module.model.objective.loss_vap(
            out["logits"], labels, reduction="mean"
        )
        vad_loss = module.model.objective.loss_vad(out["vad"], batch["vad"])
        total_loss = vap_loss + vad_loss

        if metric:
            probs = module.model.objective.get_probs(out["logits"])
            metric.update_batch(probs, batch["vad"])

        bsz = int(batch["waveform"].shape[0])
        total_examples += bsz
        total_loss_sum += float(total_loss) * bsz

    if total_examples == 0:
        raise ValueError("Validation dataloader is empty.")

    bacc: dict[str, float] = {}
    if metric:
        scores = metric.compute()
        metric.reset()
        for event_name in ["hs", "sp", "ls"]:
            if event_name not in scores:
                continue
            acc = scores[event_name]["acc"]
            bacc[event_name] = float((acc[0] + acc[1]) / 2)

    return total_loss_sum / total_examples, bacc


@hydra.main(version_base=None, config_path="conf", config_name="new_vap_home")
def main(cfg: DictConfig) -> None:
    seed = cfg.get("seed", 0)
    seed_everything(seed, workers=True)
    log.info(OmegaConf.to_yaml(cfg))

    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("Missing checkpoint_path. Example: checkpoint_path=/path/model.ckpt")

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    module = instantiate(cfg.module)
    datamodule = instantiate(cfg.datamodule)
    _load_state_dict(module, checkpoint_path)

    # This datamodule creates validation data in setup("fit").
    datamodule.prepare_data()
    datamodule.setup("fit")
    val_loader = datamodule.val_dataloader()

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    val_loss, bacc = _validation_loss_with_zero_visuals(module, val_loader, device)
    print(f"val_loss: {val_loss:.6f}")
    for event_name in ["hs", "sp", "ls"]:
        if event_name in bacc:
            print(f"bacc_{event_name}: {bacc[event_name]:.6f}")


if __name__ == "__main__":
    main()
