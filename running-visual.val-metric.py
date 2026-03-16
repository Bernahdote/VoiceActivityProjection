import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vap.data.datamodule import VAPDataset
from vap.events.events import EventConfig
from vap.metrics import VAPMetric
from vap.modules.lightning_module import VAPModule

ckpt = "/Users/willemberner/Desktop/Exjobb/checkpoints/test2-visuals/epoch=9-step=32150.ckpt"
use_val_dataset_sample = True
# Use a row from the validation split so waveform/features/vad match training-time preprocessing.
val_csv = "/Users/willemberner/datasets/splits/val_sliding.csv"
val_row_idx = 0

window_sec = 20.0
fps = 40
pixels_per_sec = 100
out_video_audio = "vap_preview_with_audio.mp4"
temp_root = Path("outputs") / "tmp_render"
video_scale_width = 240
video_frame_ext = "jpg"
video_frame_q = "4"
preview_seconds = 30.0
top_k = 5
topk_gap_rows = 1
thr = 0.5


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


def extract_event_decision_points(
    p_now: torch.Tensor,
    p_future: torch.Tensor,
    events: dict[str, list[list[tuple[int, int, int]]]],
    frame_hz_local: int,
    threshold: float,
) -> dict[str, list[dict[str, float | int | str | bool]]]:
    if p_now.ndim != 2 or p_future.ndim != 2:
        return {"hs": [], "sp": []}

    def _collect(
        event_key: str,
        use_future: bool,
        invert_hold: bool,
        target: int,
        marker: str,
        label: str,
    ) -> list[dict[str, float | int | str | bool]]:
        out = []
        event_batch0 = events.get(event_key, [[]])
        if len(event_batch0) == 0:
            return out
        for start, end, speaker in event_batch0[0]:
            start_i = int(start)
            end_i = int(end)
            if end_i <= start_i or start_i < 0:
                continue
            base = p_future if use_future else p_now
            if start_i >= base.shape[1]:
                continue
            end_i = min(end_i, base.shape[1])
            if end_i <= start_i:
                continue

            score = base[0, start_i:end_i]
            if int(speaker) == 1:
                score = 1.0 - score
            if invert_hold:
                score = 1.0 - score
            score_mean = float(score.mean().cpu())
            pred = int(score_mean >= threshold)
            out.append(
                {
                    "time_sec": (start_i + end_i) / (2.0 * frame_hz_local),
                    "score": score_mean,
                    "correct": pred == target,
                    "marker": marker,
                    "target": target,
                    "label": label,
                }
            )
        return out

    hs = []
    hs += _collect("shift", use_future=False, invert_hold=False, target=1, marker="^", label="Shift")
    hs += _collect("hold", use_future=False, invert_hold=True, target=0, marker="o", label="Hold")

    sp = []
    sp += _collect(
        "pred_shift",
        use_future=True,
        invert_hold=False,
        target=1,
        marker="^",
        label="Pre-shift",
    )
    sp += _collect(
        "pred_shift_neg",
        use_future=True,
        invert_hold=True,
        target=0,
        marker="o",
        label="Pre-hold",
    )
    return {"hs": hs, "sp": sp}


if not Path(ckpt).exists():
    raise FileNotFoundError(f"Missing file: {ckpt}")

device = "cuda" if torch.cuda.is_available() else "cpu"
# PyTorch >=2.6 defaults to weights_only=True; this checkpoint contains trusted
# OmegaConf objects, so we explicitly disable weights-only loading.
module = VAPModule.load_from_checkpoint(ckpt, map_location=device, weights_only=False)
model = module.model.to(device).eval()
if not hasattr(model, "video_dim"):
    model.video_dim = 0

sr = model.sample_rate
frame_hz = model.frame_hz
# Keep render fps aligned to model frame rate to avoid cursor/probability drift.
fps = int(frame_hz)

if not use_val_dataset_sample:
    raise ValueError("Set use_val_dataset_sample=True to match training/validation pipeline.")
if not Path(val_csv).exists():
    raise FileNotFoundError(f"Missing validation csv: {val_csv}")

dset = VAPDataset(
    path=val_csv,
    horizon=2,
    sample_rate=sr,
    frame_hz=frame_hz,
    mono=False,
    video_feature_groups=["all"],
)
if val_row_idx < 0 or val_row_idx >= len(dset):
    raise IndexError(f"val_row_idx {val_row_idx} out of range for {len(dset)} samples")

row = dset.df.iloc[val_row_idx]
sample = dset[val_row_idx]

audio_a = str(row["audio_path_a"])
audio_b = str(row["audio_path_b"])
video_a = str(Path(audio_a).with_suffix(".mp4"))
video_b = str(Path(audio_b).with_suffix(".mp4"))
for p in [audio_a, audio_b, video_a, video_b]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing file: {p}")

segment_start = float(row["start"])
segment_end = float(row["end"])
w = sample["waveform"].unsqueeze(0)  # [1, 2, T]
feat_a = sample["video_features_a"]
feat_b = sample["video_features_b"]
vad_events = sample["vad"].unsqueeze(0)
preview_sec = float(w.shape[-1]) / float(sr)
preview_seconds = preview_sec

with torch.no_grad():
    raw_out = model(
        w.to(device),
        video_features_a=feat_a.unsqueeze(0).to(device),
        video_features_b=feat_b.unsqueeze(0).to(device),
    )
    probs_train = model.objective.get_probs(raw_out["logits"])

out = {
    "probs": probs_train["probs"],
    "p_now": probs_train["p_now"],
    "p_future": probs_train["p_future"],
    "vad": raw_out["vad"].sigmoid(),
}

t_frame = torch.arange(out["p_now"].shape[1]) / frame_hz
t_audio = torch.arange(w.shape[-1]) / sr

y1 = out["p_now"][0].cpu().numpy()
y2 = out["p_future"][0].cpu().numpy()
y1_b = 1.0 - y1
y2_b = 1.0 - y2
vad = out["vad"][0].cpu().numpy()

val_metric = getattr(module, "val_metric", None)
if val_metric is None:
    val_metric = VAPMetric(
        event_config=EventConfig(
            frame_hz=frame_hz,
            max_time=int(round(float(w.shape[-1]) / float(sr))),
            equal_hold_shift=True,
        ),
        threshold=thr,
    )
metric_threshold = float(getattr(val_metric, "threshold", thr))
events = val_metric.event_extractor(vad_events)
decision_points = extract_event_decision_points(
    p_now=out["p_now"].detach().cpu(),
    p_future=out["p_future"].detach().cpu(),
    events=events,
    frame_hz_local=frame_hz,
    threshold=metric_threshold,
)
hs_decisions = decision_points["hs"]
sp_decisions = decision_points["sp"]

probs = out["probs"][0]
pred_class = probs.argmax(dim=-1)
pred_bins = model.objective.codebook.decode(pred_class).cpu().numpy()  # [T, 2, n_bins]
n_bins = pred_bins.shape[-1]
n_classes = probs.shape[-1]
top_k = min(top_k, n_classes)
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
now_end = float(sum(bin_times[:2]))
future_end = float(sum(bin_times))
ax_pnow.set_title(f"p_now: P(next speaker) in 0-{now_end:.1f}s + HS decision points")
ax_pfut.set_title(f"p_future: P(next speaker) in {now_end:.1f}-{future_end:.1f}s")
ax_vid_a.set_title("Video A")
ax_vid_b.set_title("Video B")

# Waveform A/B
color_a = "royalblue"
color_b = "orange"
wav_a = w[0, 0].cpu().numpy()
wav_b = w[0, 1].cpu().numpy()
ax_wav_a.plot(t_audio, wav_a, color=color_a, alpha=0.9)
ax_wav_b.plot(t_audio, wav_b, color=color_b, alpha=0.9)

# Add some waveform headroom so peaks do not touch the plot boundary.
wav_a_lim = max(0.05, float(np.max(np.abs(wav_a))) * 1.15)
wav_b_lim = max(0.05, float(np.max(np.abs(wav_b))) * 1.15)
ax_wav_a.set_ylim(-wav_a_lim, wav_a_lim)
ax_wav_b.set_ylim(-wav_b_lim, wav_b_lim)

# VAD probability on secondary y-axis (red)
ax_wav_a_vad = ax_wav_a.twinx()
ax_wav_b_vad = ax_wav_b.twinx()
ax_wav_a_vad.fill_between(t_frame, 0, vad[:, 0], color="red", alpha=0.25)
ax_wav_b_vad.fill_between(t_frame, 0, vad[:, 1], color="red", alpha=0.25)
ax_wav_a_vad.set_ylim(-0.05, 1.05)
ax_wav_b_vad.set_ylim(-0.05, 1.05)
ax_wav_a_vad.set_ylabel("VAD prob", color="red")
ax_wav_b_vad.set_ylabel("VAD prob", color="red")

# p_now / p_future with threshold fill
x = t_frame.cpu().numpy()
for ax, y_a, y_b in [(ax_pnow, y1, y1_b), (ax_pfut, y2, y2_b)]:
    # Show probabilities as frame-wise values (no visual interpolation between frames).
    ax.step(x, y_a, where="post", color=color_a, lw=1.4, label="P(next=A)")
    ax.step(
        x,
        y_b,
        where="post",
        color=color_b,
        lw=1.2,
        alpha=0.9,
        label="P(next=B)=1-P(next=A)",
    )
    ax.axhline(thr, color="k", ls="--", lw=1)
    ax.fill_between(
        x,
        y_a,
        thr,
        where=(y_a >= thr),
        color=color_a,
        alpha=0.18,
        step="post",
    )
    ax.fill_between(
        x,
        y_a,
        thr,
        where=(y_a < thr),
        color=color_b,
        alpha=0.18,
        step="post",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

# Hold/Shift event-level decision points on p_now (training-val metric definition).
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

    n_total = len(hs_decisions)
    n_correct = sum(int(d["correct"]) for d in hs_decisions)
    hs_acc = n_correct / n_total if n_total > 0 else 0.0
    ax_pnow.text(
        0.01,
        0.98,
        f"HS decisions: {n_correct}/{n_total} ({hs_acc:.1%}) | green correct, red wrong",
        transform=ax_pnow.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

# Shift-prediction event-level decision points on p_future (training-val metric definition).
if len(sp_decisions) > 0:
    pos_x = [d["time_sec"] for d in sp_decisions if d["target"] == 1]
    pos_y = [d["score"] for d in sp_decisions if d["target"] == 1]
    pos_c = ["#2ca02c" if d["correct"] else "#d62728" for d in sp_decisions if d["target"] == 1]
    neg_x = [d["time_sec"] for d in sp_decisions if d["target"] == 0]
    neg_y = [d["score"] for d in sp_decisions if d["target"] == 0]
    neg_c = ["#2ca02c" if d["correct"] else "#d62728" for d in sp_decisions if d["target"] == 0]

    if len(pos_x) > 0:
        ax_pfut.scatter(
            pos_x,
            pos_y,
            c=pos_c,
            marker="^",
            s=42,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="SP pre-shift decision",
        )
    if len(neg_x) > 0:
        ax_pfut.scatter(
            neg_x,
            neg_y,
            c=neg_c,
            marker="o",
            s=34,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="SP pre-hold decision",
        )

    n_total = len(sp_decisions)
    n_correct = sum(int(d["correct"]) for d in sp_decisions)
    sp_acc = n_correct / n_total if n_total > 0 else 0.0
    ax_pfut.text(
        0.01,
        0.98,
        f"SP decisions: {n_correct}/{n_total} ({sp_acc:.1%}) | green correct, red wrong",
        transform=ax_pfut.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

ax_pfut.set_xlabel("Time (s)")

# Top-k class probabilities and decoded projection windows at current frame
frame_idx0 = 0
topk_vals0, topk_idx0 = torch.topk(probs[frame_idx0], k=top_k)
topk_vals0 = (topk_vals0.cpu().numpy() * 100.0)
topk_idx0 = topk_idx0.cpu().numpy()
ypos = np.arange(top_k)
bars = ax_topk.barh(ypos, topk_vals0, color="#ef3b2c", alpha=0.95)
ax_topk.set_xlim(100, 0)
ax_topk.set_yticks(ypos)
ax_topk.set_yticklabels([str(int(i)) for i in topk_idx0])
ax_topk.yaxis.tick_right()
ax_topk.invert_yaxis()
ax_topk.set_xlabel("Top-k (%)")
ax_topk.grid(axis="x", alpha=0.25)

topk_states0 = class_states[topk_idx0]  # [k, 2, n_bins]
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
for r in range(top_k - 1):
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

# Red "now" cursor
cursor_lines = [
    ax_wav_a.axvline(0, color="red", lw=1.5),
    ax_wav_b.axvline(0, color="red", lw=1.5),
    ax_pnow.axvline(0, color="red", lw=1.5),
    ax_pfut.axvline(0, color="red", lw=1.5),
]

duration = float(t_audio[-1])
total_frames = int(math.ceil(duration * fps))
preview_sec = min(preview_seconds, duration)
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

        frame_idx = min(int(t * frame_hz), probs.shape[0] - 1)
        topk_vals, topk_idx = torch.topk(probs[frame_idx], k=top_k)
        topk_vals = (topk_vals.cpu().numpy() * 100.0)
        topk_idx = topk_idx.cpu().numpy()
        for bar, val in zip(bars, topk_vals):
            bar.set_width(float(val))
        ax_topk.set_yticklabels([str(int(ii)) for ii in topk_idx])
        topk_states = class_states[topk_idx]  # [k, 2, n_bins]
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

    # Mix WAV audio (A+B) and mux with video
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
