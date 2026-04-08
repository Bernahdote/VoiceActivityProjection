from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import torch

from vap.data.create_sliding_window_dset import sliding_window
from vap.data.datamodule import VAPDataModule
from vap.events.events import EventConfig
from vap.metrics import VAPMetric
from vap.model.vap_model import VAPModule


CKPT = "/Users/willemberner/Desktop/Exjobb/checkpoints/test2-visuals/epoch=9-step=32150.ckpt"
AUDIO_A = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.wav"
AUDIO_B = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.wav"

WINDOW_DURATION = 20.0
WINDOW_OVERLAP = 5.0
WINDOW_HORIZON = 2.0
MAX_WINDOWS = 6


def _load_vad(path: Path) -> list[list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [[float(x["start"]), float(x["end"])] for x in data.get("metadata:vad", []) if "start" in x and "end" in x]


def _build_sample_csv(tmp_dir: Path) -> tuple[Path, pd.DataFrame]:
    vad_a = _load_vad(Path(AUDIO_A).with_suffix(".json"))
    vad_b = _load_vad(Path(AUDIO_B).with_suffix(".json"))

    if not vad_a or not vad_b:
        raise RuntimeError("Missing VAD spans for the hardcoded conversation.")

    samples = sliding_window(
        vad_list=[vad_a, vad_b],
        audio_path_a=AUDIO_A,
        audio_path_b=AUDIO_B,
        duration=WINDOW_DURATION,
        overlap=WINDOW_OVERLAP,
        horizon=WINDOW_HORIZON,
    )
    if not samples:
        raise RuntimeError("No sliding windows generated for the hardcoded conversation.")
    samples = samples[:MAX_WINDOWS]

    payload = {
        "session": [],
        "audio_path_a": [],
        "audio_path_b": [],
        "start": [],
        "end": [],
        "vad_list": [],
        "dataset": [],
    }

    for sample in samples:
        payload["session"].append(sample["session"])
        payload["audio_path_a"].append(sample["audio_path_a"])
        payload["audio_path_b"].append(sample["audio_path_b"])
        payload["start"].append(sample["start"])
        payload["end"].append(sample["end"])
        payload["vad_list"].append(json.dumps(sample["vad_list"]))
        payload["dataset"].append("custom")

    df = pd.DataFrame(payload)
    out_path = tmp_dir / "sliding_window_sample.csv"
    df.to_csv(out_path, index=False)
    return out_path, df


def _build_datamodule(csv_path: Path) -> VAPDataModule:
    dm = VAPDataModule(
        test_path=str(csv_path),
        horizon=WINDOW_HORIZON,
        sample_rate=16000,
        frame_hz=50,
        mono=False,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=None,
    )
    dm.prepare_data()
    dm.setup("test")
    return dm


def _print_bacc(scores: dict[str, dict[str, torch.Tensor]]) -> None:
    hs = scores.get("hs")
    if hs is None:
        print("No hold/shift events were detected.")
        return
    acc = hs["acc"].tolist()
    hold_acc, shift_acc = acc[0], acc[1]
    bacc = (hold_acc + shift_acc) / 2.0
    print("Hold/Shift balanced accuracy:", f"{bacc:.4f}")
    print("  Hold accuracy:", f"{hold_acc:.4f}")
    print("  Shift accuracy:", f"{shift_acc:.4f}")


def main() -> None:
    for path in (CKPT, AUDIO_A, AUDIO_B, Path(AUDIO_A).with_suffix(".json"), Path(AUDIO_B).with_suffix(".json")):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building sliding-window samples for a single conversation...")
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_csv, samples = _build_sample_csv(Path(tmpdir))
        dm = _build_datamodule(sample_csv)
        print("Generated windows:", len(samples))

        module = VAPModule.load_from_checkpoint(
            checkpoint_path=CKPT,
            map_location=device,
            weights_only=False,
        )
        model = module.model.to(device).eval()

        metric = VAPMetric(EventConfig(), threshold=0.5)
        with torch.inference_mode():
            for batch in dm.test_dataloader():
                waveform = batch["waveform"].to(device)
                video_a = batch["video_features_a"].to(device)
                video_b = batch["video_features_b"].to(device)
                probs = model.probs(
                    waveform,
                    video_features_a=video_a,
                    video_features_b=video_b,
                )
                metric.update_batch(probs, batch["vad"])

        scores = metric.compute()
        _print_bacc(scores)


if __name__ == "__main__":
    main()
