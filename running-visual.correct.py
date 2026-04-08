import math
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from vap.data.create_sliding_window_dset import sliding_window
from vap.data.datamodule import VAPDataModule
from vap.events.events import EventConfig
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform
from vap.metrics import VAPMetric

CKPT = "/Users/willemberner/Desktop/Exjobb/checkpoints/test2-visuals/epoch=9-step=32150.ckpt"
AUDIO_A = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.wav"
AUDIO_B = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.wav"

WINDOW_DURATION = 20.0
WINDOW_OVERLAP = 5.0
WINDOW_HORIZON = 2.0
THRESHOLD = 0.5
MAX_WINDOWS = 6

window_sec = 20.0
fps = 40
out_video_audio = "vap_preview_with_audio.mp4"
temp_root = Path("outputs") / "tmp_render"
video_scale_width = 240
video_frame_ext = "jpg"
video_frame_q = "4"
preview_seconds = 30.0
top_k = 5
topk_gap_rows = 1


def run_ffmpeg(args):
    subprocess.run(
        args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_topk_window_rgb(topk_states, gray_rgb, blue_rgb, orange_rgb, gap_rows):
    k = topk_states.shape[0]
    n_bins_local = topk_states.shape[-1]
    stride = 2 + gap_rows
    total_rows = (k * 2) + (max(0, k - 1) * gap_rows)

    window_rgb = np.tile(gray_rgb, (total_rows, n_bins_local, 1))
    row_offsets = np.zeros(k, dtype=np.int32)
    for r in range(k):
        row0 = r * stride
        row_offsets[r] = row0
        window_rgb[row0, topk_states[r, 0] > 0.5] = blue_rgb
        window_rgb[row0 + 1, topk_states[r, 1] > 0.5] = orange_rgb
    return window_rgb, row_offsets


def build_sample_csv(tmp_dir: Path) -> tuple[Path, pd.DataFrame]:
    vad_a = [[x["start"], x["end"]] for x in json.load(open(AUDIO_A.replace(".wav", ".json"))).get("metadata:vad", [])]
    vad_b = [[x["start"], x["end"]] for x in json.load(open(AUDIO_B.replace(".wav", ".json"))).get("metadata:vad", [])]

    samples = sliding_window(
        vad_list=[vad_a, vad_b],
        audio_path_a=AUDIO_A,
        audio_path_b=AUDIO_B,
        duration=WINDOW_DURATION,
        overlap=WINDOW_OVERLAP,
        horizon=WINDOW_HORIZON,
    )
    if not samples:
        raise RuntimeError("No sliding windows generated.")
    samples = samples[:MAX_WINDOWS]
    if not samples:
        raise RuntimeError("No sliding windows available after limiting to the first windows.")
    payload = {"session": [], "audio_path_a": [], "audio_path_b": [], "start": [], "end": [], "vad_list": [], "dataset": []}
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


def collect_event_decisions(
    events: dict[str, list[list[tuple[int, int, int]]]],
    p_now: torch.Tensor,
    p_future: torch.Tensor,
    frame_hz_local: int,
    window_start: float,
    display_frame_start: int,
    threshold: float,
    segment_start: float,
) -> tuple[dict[str, list[dict[str, float | int | str | bool]]], dict[str, int]]:
    """
    Mirror the validation plotting: gather HS and SP event-level decisions.
    Times are shifted so 0 corresponds to the earliest window start (segment_start).
    Overlapping frames before display_frame_start are skipped to avoid double-counting.
    """
    decisions = {"hs": [], "sp": []}
    stats = {"hs_total": 0, "hs_correct": 0, "sp_total": 0, "sp_correct": 0}

    if p_now.ndim != 2 or p_future.ndim != 2:
        return decisions, stats

    def _collect(event_key: str, use_future: bool, invert_hold: bool, target: int, marker: str, label: str):
        base = p_future if use_future else p_now
        event_batches = events.get(event_key) or []
        if len(event_batches) == 0:
            return []
        out_list = []
        for start, end, speaker in event_batches[0]:
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i or start_i < 0:
                continue
            if start_i >= base.shape[1]:
                continue
            end_i = min(end_i, base.shape[1])
            if end_i <= start_i:
                continue

            # Drop the overlapping prefix so we only plot each frame once.
            disp_start = max(start_i, display_frame_start)
            disp_end = min(end_i, base.shape[1])
            if disp_end <= disp_start:
                continue

            score = base[0, disp_start:disp_end]
            if int(speaker) == 1:
                score = 1.0 - score
            if invert_hold:
                score = 1.0 - score

            score_mean = float(score.mean().cpu())
            pred = int(score_mean >= threshold)
            center_frame = (disp_start + disp_end) / 2.0
            time_rel = (window_start - segment_start) + center_frame / frame_hz_local

            is_hs = event_key in ("shift", "hold")
            stats_key = "hs" if is_hs else "sp"
            stats[f"{stats_key}_total"] += 1
            stats[f"{stats_key}_correct"] += int(pred == target)

            out_list.append(
                {
                    "time_sec": time_rel,
                    "score": score_mean,
                    "correct": pred == target,
                    "marker": marker,
                    "target": target,
                    "label": label,
                }
            )
        return out_list

    decisions["hs"] += _collect("shift", use_future=False, invert_hold=False, target=1, marker="^", label="Shift")
    decisions["hs"] += _collect("hold", use_future=False, invert_hold=True, target=0, marker="o", label="Hold")

    decisions["sp"] += _collect("pred_shift", use_future=True, invert_hold=False, target=1, marker="^", label="Pre-shift")
    decisions["sp"] += _collect("pred_shift_neg", use_future=True, invert_hold=True, target=0, marker="o", label="Pre-hold")

    return decisions, stats


def inference() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, df = build_sample_csv(Path(tmpdir))
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
        dm.setup(stage="test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    module = VAPModule.load_from_checkpoint(checkpoint_path=CKPT, map_location=device, weights_only=False)
    model = module.model.to(device).eval()

    metric = VAPMetric(EventConfig(), threshold=0.5)
    hs_decisions: list[dict[str, float | int | str | bool]] = []
    sp_decisions: list[dict[str, float | int | str | bool]] = []
    decision_stats_total: dict[str, int] = {
        "hs_total": 0,
        "hs_correct": 0,
        "sp_total": 0,
        "sp_correct": 0,
    }

    aggregated: dict[str, torch.Tensor] = {}
    step_frames = int(round((WINDOW_DURATION - WINDOW_OVERLAP) * model.frame_hz))
    step_frames = max(1, step_frames)
    test_df = dm.test_dset.df.reset_index(drop=True)
    segment_start = float(test_df["start"].min())

    with torch.no_grad():
        for idx, batch in enumerate(dm.test_dataloader()):
            waveform = batch["waveform"].to(device)
            video_a = batch["video_features_a"].to(device)
            video_b = batch["video_features_b"].to(device)
            out = model.probs(
                waveform,
                video_features_a=video_a,
                video_features_b=video_b,
            )
            events = metric.event_extractor(batch["vad"])
            preds, targets = metric.extract_prediction_and_targets(
                p_now=out["p_now"], p_fut=out["p_future"], events=events
            )
            metric._update_metrics(preds, targets)
            window_len = out["p_now"].shape[1]
            display_frame_start = 0 if idx == 0 else max(0, window_len - step_frames)
            sample_start = float(test_df.iloc[idx]["start"])
            batch_decisions, batch_stats = collect_event_decisions(
                events=events,
                p_now=out["p_now"],
                p_future=out["p_future"],
                frame_hz_local=model.frame_hz,
                window_start=sample_start,
                display_frame_start=display_frame_start,
                threshold=THRESHOLD,
                segment_start=segment_start,
            )
            hs_decisions.extend(batch_decisions["hs"])
            sp_decisions.extend(batch_decisions["sp"])
            for key in decision_stats_total:
                decision_stats_total[key] += batch_stats.get(key, 0)
            if not aggregated:
                aggregated = {k: v.clone() for k, v in out.items()}
            else:
                for key in ("p_now", "p_future", "probs", "vad", "H"):
                    if key not in aggregated or key not in out:
                        continue
                    aggregated[key] = torch.cat(
                        [aggregated[key], out[key][:, -step_frames:]], dim=1
                    )

    scores = metric.compute()

    return {
        "out": aggregated,
        "model": model,
        "samples_df": df,
        "frame_hz": model.frame_hz,
        "metrics": scores,
        "hs_decisions": hs_decisions,
        "sp_decisions": sp_decisions,
        "decision_stats": decision_stats_total,
        "segment_start": segment_start,
    }


if __name__ == "__main__":
    result = inference()
    out = result["out"]
    model = result["model"]
    frame_hz = result["frame_hz"]
    samples_df = result["samples_df"]
    metrics = result.get("metrics")
    hs_metrics = metrics.get("hs") if metrics else None
    if hs_metrics and hs_metrics.get("acc") is not None:
        acc_vals = hs_metrics["acc"].tolist()
        hold_acc_v = acc_vals[0]
        shift_acc_v = acc_vals[1]
        bacc_v = (hold_acc_v + shift_acc_v) / 2.0
        print("VAPMetric hold/shift:")
        print(f"  Hold accuracy: {hold_acc_v:.4f}")
        print(f"  Shift accuracy: {shift_acc_v:.4f}")
        print(f"  Balanced accuracy: {bacc_v:.4f}")

    audio_a = AUDIO_A
    audio_b = AUDIO_B
    video_a = str(Path(audio_a).with_suffix(".mp4"))
    video_b = str(Path(audio_b).with_suffix(".mp4"))

    for p in (CKPT, audio_a, audio_b, video_a, video_b):
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing file: {p}")

    segment_start = float(samples_df["start"].min())
    segment_end = float(samples_df["end"].max())
    segment_duration = max(0.0, segment_end - segment_start)
    preview_sec = min(preview_seconds, segment_duration)

    sr = model.sample_rate
    w1, _ = load_waveform(
        audio_a,
        sample_rate=sr,
        mono=True,
        start_time=segment_start,
        end_time=segment_end,
    )
    w2, _ = load_waveform(
        audio_b,
        sample_rate=sr,
        mono=True,
        start_time=segment_start,
        end_time=segment_end,
    )
    n_samples = min(w1.shape[-1], w2.shape[-1])
    if n_samples == 0:
        raise RuntimeError("Loaded empty audio segment.")
    w = torch.cat([w1[..., :n_samples], w2[..., :n_samples]], dim=0)
    t_audio = torch.arange(w.shape[-1]) / sr

    t_frame = torch.arange(out["p_now"].shape[1]) / frame_hz
    y1 = out["p_now"][0].cpu().numpy()
    y2 = out["p_future"][0].cpu().numpy()
    vad = out["vad"][0].cpu().numpy()
    hs_decisions = result["hs_decisions"]
    sp_decisions = result["sp_decisions"]
    decision_stats = result.get("decision_stats", {})
    probs = out["probs"][0]
    pred_class = probs.argmax(dim=-1)
    pred_bins = model.objective.codebook.decode(pred_class).cpu().numpy()  # [T, 2, n_bins]
    n_bins = pred_bins.shape[-1]
    n_classes = probs.shape[-1]
    top_k_local = min(top_k, n_classes)
    class_states = model.objective.codebook.decode(torch.arange(n_classes)).cpu().numpy()
    gray_rgb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    orange_rgb = np.array([0.95, 0.72, 0.42], dtype=np.float32)
    blue_rgb = np.array([0.42, 0.64, 0.80], dtype=np.float32)

    fig = plt.figure(figsize=(18, 10))
    fig.subplots_adjust(left=0.04, right=0.99, top=0.98, bottom=0.06, hspace=0.35, wspace=0.2)
    gs = fig.add_gridspec(
        6, 3, height_ratios=[1, 1, 1, 1.2, 0.9, 0.9], width_ratios=[1, 1, 1.3]
    )

    ax_vid_a = fig.add_subplot(gs[:, 0])
    ax_vid_b = fig.add_subplot(gs[:, 1])

    ax_wav_a = fig.add_subplot(gs[0, 2])
    ax_wav_b = fig.add_subplot(gs[1, 2], sharex=ax_wav_a)
    ax_pnow = fig.add_subplot(gs[2, 2], sharex=ax_wav_a)
    ax_pfut = fig.add_subplot(gs[3, 2], sharex=ax_wav_a)
    gs_bottom = gs[4:, 2].subgridspec(1, 2, width_ratios=[1.0, 0.95], wspace=0.12)
    ax_topk = fig.add_subplot(gs_bottom[0, 0])
    ax_topk_windows = fig.add_subplot(gs_bottom[0, 1])

    ax_wav_a.set_title("Waveform A")
    ax_wav_b.set_title("Waveform B")
    bin_times = list(getattr(model.objective, "bin_times", [0.2, 0.4, 0.6, 0.8]))
    ax_pnow.set_title("p_now + HS decision points")
    ax_pfut.set_title("p_future")
    ax_vid_a.set_title("Video A")
    ax_vid_b.set_title("Video B")

    color_a = "royalblue"
    color_b = "orange"
    wav_a = w[0].cpu().numpy()
    wav_b = w[1].cpu().numpy()
    ax_wav_a.plot(t_audio, wav_a, color=color_a, alpha=0.9)
    ax_wav_b.plot(t_audio, wav_b, color=color_b, alpha=0.9)

    wav_a_lim = max(0.05, float(np.max(np.abs(wav_a))) * 1.15)
    wav_b_lim = max(0.05, float(np.max(np.abs(wav_b))) * 1.15)
    ax_wav_a.set_ylim(-wav_a_lim, wav_a_lim)
    ax_wav_b.set_ylim(-wav_b_lim, wav_b_lim)

    ax_wav_a_vad = ax_wav_a.twinx()
    ax_wav_b_vad = ax_wav_b.twinx()
    ax_wav_a_vad.fill_between(t_frame, 0, vad[:, 0], color="red", alpha=0.25)
    ax_wav_b_vad.fill_between(t_frame, 0, vad[:, 1], color="red", alpha=0.25)
    ax_wav_a_vad.set_ylim(-0.05, 1.05)
    ax_wav_b_vad.set_ylim(-0.05, 1.05)
    ax_wav_a_vad.set_ylabel("VAD prob", color="red")
    ax_wav_b_vad.set_ylabel("VAD prob", color="red")

    x = t_frame.cpu().numpy()
    for ax, y in [(ax_pnow, y1), (ax_pfut, y2)]:
        ax.step(x, y, where="post", color=color_a, lw=1.4) 
        ax.axhline(THRESHOLD, color="k", ls="--", lw=1)
        ax.fill_between(
            x,
            y,
            THRESHOLD,
            where=(y >= THRESHOLD),
            color=color_a,
            alpha=0.18,
            step="post",
        )
        ax.fill_between(
            x,
            y,
            THRESHOLD,
            where=(y < THRESHOLD),
            color=color_b,
            alpha=0.18,
            step="post",
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    if len(hs_decisions) > 0:
        shift_x = [d["time_sec"] for d in hs_decisions if d["target"] == 1]
        shift_y = [d["score"] for d in hs_decisions if d["target"] == 1]
        shift_c = ["#2ca02c" if d["correct"] else "#d62728" for d in hs_decisions if d["target"] == 1]
        hold_x = [d["time_sec"] for d in hs_decisions if d["target"] == 0]
        hold_y = [d["score"] for d in hs_decisions if d["target"] == 0]
        hold_c = ["#2ca02c" if d["correct"] else "#d62728" for d in hs_decisions if d["target"] == 0]

        if len(shift_x) > 0:
            ax_pnow.scatter(
                shift_x,
                shift_y,
                c=shift_c,
                marker="^",
                s=42,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
                label="HS shift decision",
            )
        if len(hold_x) > 0:
            ax_pnow.scatter(
                hold_x,
                hold_y,
                c=hold_c,
                marker="o",
                s=34,
                edgecolors="black",
                linewidths=0.5,
                zorder=6,
                label="HS hold decision",
            )

        hs_total = decision_stats.get("hs_total", len(hs_decisions))
        hs_correct = decision_stats.get(
            "hs_correct", sum(int(d.get("correct", False)) for d in hs_decisions)
        )
        hs_acc = hs_correct / hs_total if hs_total > 0 else 0.0

    # SP decision plotting removed per request

    ax_pfut.set_xlabel("Time (s)")

    frame_idx0 = 0
    topk_vals0, topk_idx0 = torch.topk(probs[frame_idx0], k=top_k_local)
    topk_vals0 = (topk_vals0.cpu().numpy() * 100.0)
    topk_idx0 = topk_idx0.cpu().numpy()
    ypos = np.arange(top_k_local)
    bars = ax_topk.barh(ypos, topk_vals0, color="#ef3b2c", alpha=0.95)
    ax_topk.set_xlim(100, 0)
    ax_topk.set_yticks(ypos)
    ax_topk.set_yticklabels([str(int(i)) for i in topk_idx0])
    ax_topk.yaxis.tick_right()
    ax_topk.invert_yaxis()
    ax_topk.set_xlabel("Top-k (%)")
    ax_topk.grid(axis="x", alpha=0.25)

    topk_states0 = class_states[topk_idx0]
    topk_window_rgb, topk_row_offsets = build_topk_window_rgb(
        topk_states0,
        gray_rgb,
        blue_rgb,
        orange_rgb,
        topk_gap_rows,
    )

    im_topk = ax_topk_windows.imshow(
        topk_window_rgb,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    ax_topk_windows.set_xlabel("Top-k Projection Windows")
    ax_topk_windows.set_yticks(topk_row_offsets + 0.5)
    ax_topk_windows.set_yticklabels([])
    ax_topk_windows.set_xticks(np.arange(n_bins))
    ax_topk_windows.set_xticklabels([f"{i+1}" for i in range(n_bins)], fontsize=9)
    ax_topk_windows.tick_params(axis="x", rotation=0)
    ax_topk_windows.set_xticks(np.arange(-0.5, n_bins, 1), minor=True)
    ax_topk_windows.grid(which="minor", axis="x", color="black", linewidth=1.2, alpha=1.0)
    ax_topk_windows.tick_params(which="minor", bottom=False, left=False)
    for row0 in topk_row_offsets:
        ax_topk_windows.axhline(row0 - 0.5, color="black", lw=1.2)
        ax_topk_windows.axhline(row0 + 0.5, color="black", lw=1.2)
        ax_topk_windows.axhline(row0 + 1.5, color="black", lw=1.2)
    for r in range(top_k_local - 1):
        gap_start = topk_row_offsets[r] + 2
        ax_topk_windows.axhspan(
            gap_start - 0.5,
            gap_start + topk_gap_rows - 0.5,
            facecolor="white",
            edgecolor="none",
            zorder=3,
        )
    for spine in ax_topk_windows.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.4)
        spine.set_color("black")
    ax_topk_windows.text(-0.9, 0, "A", ha="center", va="center", fontsize=9)
    ax_topk_windows.text(-0.9, 1, "B", ha="center", va="center", fontsize=9)

    for ax in (
        ax_wav_a,
        ax_wav_b,
        ax_pnow,
        ax_pfut,
        ax_wav_a_vad,
        ax_wav_b_vad,
    ):
        ax.margins(x=0)
        ax.set_xlim(0, float(t_audio[-1]))

    plt.tight_layout()

    cursor_lines = [
        ax_wav_a.axvline(0, color="red", lw=1.5),
        ax_wav_b.axvline(0, color="red", lw=1.5),
        ax_pnow.axvline(0, color="red", lw=1.5),
        ax_pfut.axvline(0, color="red", lw=1.5),
    ]

    duration = float(t_audio[-1])
    total_frames = int(math.ceil(duration * fps))
    total_frames = min(total_frames, int(math.ceil(preview_sec * fps)))
    half_win = window_sec / 2.0

    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temp_root) as tmpdir:
        print(f"[info] temp dir: {tmpdir}", flush=True)
        frames_dir = Path(tmpdir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        video_frames_a = Path(tmpdir) / "video_a_frames"
        video_frames_b = Path(tmpdir) / "video_b_frames"
        video_frames_a.mkdir(parents=True, exist_ok=True)
        video_frames_b.mkdir(parents=True, exist_ok=True)

        print("[stage] extract video frames A", flush=True)
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(segment_start),
                "-i",
                video_a,
                "-vf",
                f"fps={fps},scale={video_scale_width}:-2",
                "-q:v",
                video_frame_q,
                "-t",
                str(preview_sec),
                str(video_frames_a / f"frame_%06d.{video_frame_ext}"),
            ],
        )
        print("[stage] extract video frames B", flush=True)
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(segment_start),
                "-i",
                video_b,
                "-vf",
                f"fps={fps},scale={video_scale_width}:-2",
                "-q:v",
                video_frame_q,
                "-t",
                str(preview_sec),
                str(video_frames_b / f"frame_%06d.{video_frame_ext}"),
            ],
        )

        frames_a = sorted(video_frames_a.glob(f"frame_*.{video_frame_ext}"))
        frames_b = sorted(video_frames_b.glob(f"frame_*.{video_frame_ext}"))
        if not frames_a or not frames_b:
            raise RuntimeError("No video frames extracted. Check video paths.")

        img_a = plt.imread(frames_a[0])
        img_b = plt.imread(frames_b[0])
        im_a = ax_vid_a.imshow(img_a)
        im_b = ax_vid_b.imshow(img_b)
        ax_vid_a.axis("off")
        ax_vid_b.axis("off")

        print("[stage] render combined frames", flush=True)
        for i in range(total_frames):
            t = i / fps
            left = t - half_win
            right = t + half_win
            for ax in (ax_wav_a, ax_wav_b, ax_pnow, ax_pfut):
                ax.set_xlim(left, right)
            for line in cursor_lines:
                line.set_xdata([t, t])

            frame_idx = min(int(round(t * frame_hz)), probs.shape[0] - 1)
            topk_vals, topk_idx = torch.topk(probs[frame_idx], k=top_k_local)
            topk_vals = (topk_vals.cpu().numpy() * 100.0)
            topk_idx = topk_idx.cpu().numpy()
            for bar, val in zip(bars, topk_vals):
                bar.set_width(float(val))
            ax_topk.set_yticklabels([str(int(ii)) for ii in topk_idx])
            topk_states = class_states[topk_idx]
            topk_window_rgb, _ = build_topk_window_rgb(
                topk_states,
                gray_rgb,
                blue_rgb,
                orange_rgb,
                topk_gap_rows,
            )
            im_topk.set_data(topk_window_rgb)

            idx_a = min(i, len(frames_a) - 1)
            idx_b = min(i, len(frames_b) - 1)
            im_a.set_data(plt.imread(frames_a[idx_a]))
            im_b.set_data(plt.imread(frames_b[idx_b]))

            frame_path = frames_dir / f"frame_{i:06d}.png"
            fig.savefig(frame_path, dpi=70)
            if i % 300 == 0 and i != 0:
                print(f"[render] {i}/{total_frames}", flush=True)

        print("[stage] mix audio", flush=True)
        mixed_audio = Path(tmpdir) / "mixed_audio.m4a"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(segment_start),
                "-t",
                str(preview_sec),
                "-i",
                audio_a,
                "-ss",
                str(segment_start),
                "-t",
                str(preview_sec),
                "-i",
                audio_b,
                "-filter_complex",
                "amix=inputs=2:duration=longest",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(mixed_audio),
            ],
        )

        print("[stage] mux final video", flush=True)
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-r",
                str(fps),
                "-i",
                str(frames_dir / "frame_%06d.png"),
                "-i",
                str(mixed_audio),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                out_video_audio,
            ],
        )
