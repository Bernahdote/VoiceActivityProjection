from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

FAU_NAMES = [
    "InnerBrowRaiser", "OuterBrowRaiser", "BrowLowerer",
    "UpperLidRaiser", "CheekRaiser", "LidTightener", "NoseWrinkler", "UpperLipRaiser",
    "LipCornerPuller", "CheekPuffer", "Dimpler",
    "LipCornerDepressor", "LowerLipDepressor", "ChinRaiser", "LipPuckerer",
    "LipStretcher", "LipFunneler", "LipTightener", "LipPressor", "LipsPart",
    "JawDrop", "LipSuck", "JawSideways", "EyesClosed" 
]


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


def _evaluate_with_mask(module, loader, device, zeroed_dim: int | None = None) -> float:
    total_examples = 0
    loss_sum = 0.0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"dim={zeroed_dim}", leave=False):
            batch = _to_device(batch, device)

            if zeroed_dim is not None:
                batch["video_features_a"] = batch["video_features_a"].clone()
                batch["video_features_b"] = batch["video_features_b"].clone()
                batch["video_features_a"][..., zeroed_dim] = 0.0
                batch["video_features_b"][..., zeroed_dim] = 0.0

            out = module.model(
                batch["waveform"],
                video_features_a=batch["video_features_a"],
                video_features_b=batch["video_features_b"],
            )
            labels = module.model.extract_labels(batch["vad"])
            loss = module.model.objective.loss_vap(out["logits"], labels, reduction="mean")
            va_loss = module.model.objective.loss_vad(out["vad"], batch["vad"])

            bsz = int(batch["waveform"].shape[0])
            total_examples += bsz
            loss_sum += (float(loss) + float(va_loss)) * bsz

    return loss_sum / total_examples


CHECKPOINT = "./runs_new/VAP_debug/9n7fohs7/checkpoints/epoch=11-step=38580.ckpt"
TEST_CSV = "/mnt/sdb/willem/datasets/splits/test_sliding.csv"


@hydra.main(version_base=None, config_path="vap/conf", config_name="evaluate")
def main(cfg_eval: DictConfig) -> None:
    checkpoint_path = Path(to_absolute_path(CHECKPOINT))
    test_csv_path = Path(to_absolute_path(TEST_CSV))

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Test CSV not found: {test_csv_path}")

    cfg_eval.module.model.video_dim = 24
    module = instantiate(cfg_eval.module)
    _load_checkpoint(module, checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = module.to(device)
    module.eval()

    cfg_eval.datamodule.test_path = str(test_csv_path)
    cfg_eval.datamodule.video_feature_groups = ["fauv"]
    datamodule = instantiate(cfg_eval.datamodule)
    datamodule.prepare_data()
    datamodule.setup("test")
    loader = datamodule.test_dataloader()

    print(f"checkpoint: {checkpoint_path}")
    print(f"test_csv:   {test_csv_path}")
    print(f"device:     {device}\n")

    baseline = _evaluate_with_mask(module, loader, device, zeroed_dim=None)
    print(f"Baseline loss: {baseline:.6f}\n")

    results = []
    n_dims = 24
    for i in range(n_dims):
        loss = _evaluate_with_mask(module, loader, device, zeroed_dim=i)
        delta = loss - baseline
        name = FAU_NAMES[i] if i < len(FAU_NAMES) else f"dim_{i}"
        results.append((i, name, loss, delta))
        print(f"[{i:2d}] {name:<22s}  loss={loss:.6f}  delta={delta:+.6f}")

    print("\n--- Ranked by importance (largest delta first) ---")
    for i, name, loss, delta in sorted(results, key=lambda x: -x[3]):
        print(f"[{i:2d}] {name:<22s}  loss={loss:.6f}  delta={delta:+.6f}")


if __name__ == "__main__":
    main()
