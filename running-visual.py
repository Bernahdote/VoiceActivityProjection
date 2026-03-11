import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform

ckpt = "/Users/willemberner/Desktop/Exjobb/epoch=7-step=14008.ckpt"
audio_a = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.wav"
audio_b = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.wav"
video_a = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.mp4"
video_b = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.mp4"

start_time = 0.0
window_sec = 20.0
fps = 40
pixels_per_sec = 100
out_video_audio = "vap_preview_with_audio.mp4"
temp_root = Path("outputs") / "tmp_render"
video_scale_width = 240
video_frame_ext = "jpg"
video_frame_q = "4"
preview_seconds = 60.0
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


for p in [ckpt, audio_a, audio_b, video_a, video_b]:
    if not Path(p).exists():
        raise FileNotFoundError(f"Missing file: {p}")

device = "cuda" if torch.cuda.is_available() else "cpu"
module = VAPModule.load_from_checkpoint(ckpt, map_location=device)
model = module.model.to(device).eval()
if not hasattr(model, "video_dim"):
    model.video_dim = 0

sr = model.sample_rate
frame_hz = model.frame_hz

segment_start = start_time
segment_end_req = segment_start + preview_seconds

w1, _ = load_waveform(
    audio_a,
    sample_rate=sr,
    mono=True,
    start_time=segment_start,
    end_time=segment_end_req,
)
w2, _ = load_waveform(
    audio_b,
    sample_rate=sr,
    mono=True,
    start_time=segment_start,
    end_time=segment_end_req,
)

n_samples = min(w1.shape[-1], w2.shape[-1])
if n_samples == 0:
    raise RuntimeError("Loaded empty audio segment.")
w1 = w1[..., :n_samples]
w2 = w2[..., :n_samples]
w = torch.cat([w1, w2], dim=0).unsqueeze(0)  # [1, 2, T]

preview_sec = n_samples / sr

with torch.no_grad():
    out = model.probs(w.to(device))

t_frame = torch.arange(out["p_now"].shape[1]) / frame_hz
t_audio = torch.arange(w.shape[-1]) / sr

y1 = out["p_now"][0].cpu().numpy()
y2 = out["p_future"][0].cpu().numpy()
vad = out["vad"][0].cpu().numpy()
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
ax_pnow.set_title("p_now")
ax_pfut.set_title("p_future")
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
for ax, y, title in [(ax_pnow, y1, "p_now (A)"), (ax_pfut, y2, "p_future (A)")]:
    ax.plot(x, y, color="black", lw=1)
    ax.axhline(thr, color="k", ls="--", lw=1)
    ax.fill_between(x, y, thr, where=(y >= thr), color=color_a, alpha=0.3, interpolate=True)
    ax.fill_between(x, y, thr, where=(y < thr), color=color_b, alpha=0.3, interpolate=True)
    ax.set_ylim(-0.05, 1.05)

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

        frame_idx = min(int(round(t * frame_hz)), probs.shape[0] - 1)
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
