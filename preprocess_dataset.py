"""
Preprocess dataset into single .pt files for fast data loading.

Performs all I/O-heavy operations once (wav loading, npz loading, interpolation)
and saves each sample as a single .pt file.

Usage:
    uv run preprocess_dataset.py \
        --csv data/splits/dev/dev_train_slide.csv \
        --output_dir data/preprocessed/dev_train \
        --video_feature_groups body_pose left_hand_pose right_hand_pose head gaze alignment_head_rotation fauv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from vap.data.datamodule import (
    load_df,
    force_correct_nsamples,
    VIDEO_FEATURE_SLICES,
)
from vap.utils.audio import load_waveform
from vap.utils.utils import vad_list_to_onehot


def compute_delta_features(feats: torch.Tensor, window: int = 10) -> torch.Tensor:
    """Delta = current frame - mean of previous `window` frames. Zero-padded for first frames."""
    x = feats.T.unsqueeze(0)                          # (1, C, T)
    kernel = torch.ones(1, 1, window, device=feats.device) / window
    avg = F.conv1d(F.pad(x, (window, 0)), kernel.expand(x.shape[1], -1, -1), groups=x.shape[1])
    avg = avg[..., :-1]                                # shift: use frames [t-window, t-1], not [t-window+1, t]
    avg = avg.squeeze(0).T                             # (T, C)
    delta = feats - avg
    delta[:window] = 0.0
    return delta


def select_video_features(feats: torch.Tensor, groups: list[str]) -> torch.Tensor:
    if groups == ["all"]:
        return feats
    selected = [feats[:, VIDEO_FEATURE_SLICES[g]] for g in groups]
    return torch.cat(selected, dim=-1)


def preprocess_sample(
    row,
    sample_rate: int,
    frame_hz: int,
    horizon: float,
    video_feature_groups: list[str],
) -> dict:
    dur = round(row["end"] - row["start"])
    n_samples = int(dur * sample_rate)

    wa, _ = load_waveform(
        row["audio_path_a"],
        start_time=row["start"],
        end_time=row["end"],
        sample_rate=sample_rate,
        mono=True,
    )
    wb, _ = load_waveform(
        row["audio_path_b"],
        start_time=row["start"],
        end_time=row["end"],
        sample_rate=sample_rate,
        mono=True,
    )
    w = torch.cat([wa, wb], dim=0)
    w = force_correct_nsamples(w, n_samples)

    video_path_a = str(Path(row["audio_path_a"]).with_suffix(".f.npz"))
    video_path_b = str(Path(row["audio_path_b"]).with_suffix(".f.npz"))

    za = np.load(video_path_a, allow_pickle=False)
    zb = np.load(video_path_b, allow_pickle=False)
    fa = torch.from_numpy(za["features"]).float()
    fb = torch.from_numpy(zb["features"]).float()
    fa = select_video_features(fa, video_feature_groups)
    fb = select_video_features(fb, video_feature_groups)

    src_fps = 30.0
    start_idx = int(row["start"] * src_fps)
    end_idx = int(row["end"] * src_fps)

    fa = fa[start_idx:end_idx]
    fb = fb[start_idx:end_idx]

    window_dur = float(row["end"] - row["start"])
    expected_src_len = max(1, int(round(window_dur * src_fps)))
    target_len = max(1, int(round(window_dur * frame_hz)))

    if fa.shape[0] == 0:
        fa = torch.zeros((expected_src_len, fa.shape[1]), dtype=fa.dtype)
    if fb.shape[0] == 0:
        fb = torch.zeros((expected_src_len, fb.shape[1]), dtype=fb.dtype)

    fa = fa[:expected_src_len]
    fb = fb[:expected_src_len]

    if fa.shape[0] < expected_src_len:
        pad_n = expected_src_len - fa.shape[0]
        fa = torch.cat([fa, fa[-1:].repeat(pad_n, 1)], dim=0)
    if fb.shape[0] < expected_src_len:
        pad_n = expected_src_len - fb.shape[0]
        fb = torch.cat([fb, fb[-1:].repeat(pad_n, 1)], dim=0)

    fa = torch.nn.functional.interpolate(
        fa.T.unsqueeze(0), size=target_len, mode="linear", align_corners=False
    ).squeeze(0).T
    fb = torch.nn.functional.interpolate(
        fb.T.unsqueeze(0), size=target_len, mode="linear", align_corners=False
    ).squeeze(0).T

    fa = torch.cat([fa, compute_delta_features(fa)], dim=-1)
    fb = torch.cat([fb, compute_delta_features(fb)], dim=-1)

    vad = vad_list_to_onehot(
        row["vad_list"], duration=dur + horizon, frame_hz=frame_hz
    )

    return {
        "session": row.get("session", ""),
        "dataset": row.get("dataset", ""),
        "waveform": w,
        "vad": vad,
        "video_features_a": fa,
        "video_features_b": fb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the sliding window CSV")
    parser.add_argument("--output_dir", required=True, help="Directory to save .pt files")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--frame_hz", type=int, default=50)
    parser.add_argument("--horizon", type=float, default=2.0)
    parser.add_argument(
        "--video_feature_groups",
        nargs="+",
        default=["body_pose", "left_hand_pose", "right_hand_pose", "head", "gaze", "alignment_head_rotation", "fauv"],
    )
    args = parser.parse_args()

    df = load_df(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    errors = []
    for idx in tqdm(range(len(df)), desc=f"Preprocessing -> {args.output_dir}"):
        row = df.iloc[idx]
        try:
            sample = preprocess_sample(
                row,
                sample_rate=args.sample_rate,
                frame_hz=args.frame_hz,
                horizon=args.horizon,
                video_feature_groups=args.video_feature_groups,
            )
            torch.save(sample, output_dir / f"{idx:06d}.pt")
        except Exception as e:
            errors.append((idx, str(e)))
            print(f"Error at index {idx}: {e}")

    print(f"\nDone. {len(df) - len(errors)}/{len(df)} samples saved to {output_dir}")
    if errors:
        print(f"{len(errors)} errors:")
        for idx, msg in errors:
            print(f"  [{idx}] {msg}")


if __name__ == "__main__":
    main()
