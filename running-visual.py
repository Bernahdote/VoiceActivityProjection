
import torch
from vap.utils.audio import load_waveform
from vap.model.vap_model import VAPModule
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pathlib import Path
import tempfile
import math

ckpt = "/Users/willemberner/Desktop/Exjobb/epoch=7-step=14008.ckpt"
audio_a = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0018/V00_S0696_I00000544_P0844A.wav"
audio_b = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0027/V00_S0696_I00000544_P0847.wav"
video_a = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0018/V00_S0696_I00000544_P0844A.mp4" 
video_b = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0027/V00_S0696_I00000544_P0847.mp4"


window_sec = 20.0
fps = 40
pixels_per_sec = 100
out_video_audio = "vap_preview_with_audio.mp4"
temp_root = Path("outputs") / "tmp_render"
video_scale_width = 240
video_frame_ext = "jpg"
video_frame_q = "4"
preview_seconds = 60.0


def run_ffmpeg(args):
    subprocess.run(
        args,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


w1, _ = load_waveform(
    audio_a, sample_rate=16000, mono=True, start_time=0, end_time=preview_seconds
)
w2, _ = load_waveform(
    audio_b, sample_rate=16000, mono=True, start_time=0, end_time=preview_seconds
)


w = torch.cat([w1, w2], dim=0).unsqueeze(0)  # [1, 2, T]

module = VAPModule.load_from_checkpoint(ckpt)
model = module.model.eval()

with torch.no_grad():
    out = model.probs(w)

frame_hz = 50  # model frame rate
sr = 16000
thr = 0.5
vad_thr = 0.5

t_frame = torch.arange(out["p_now"].shape[1]) / frame_hz
t_audio = torch.arange(w.shape[-1]) / sr

y1 = out["p_now"][0].cpu().numpy()
y2 = out["p_future"][0].cpu().numpy()
vad = out["vad"][0].cpu().numpy()

fig = plt.figure(figsize=(18, 9))
fig.subplots_adjust(left=0.04, right=0.99, top=0.98, bottom=0.06, hspace=0.35, wspace=0.2)
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2], width_ratios=[1, 1, 1.3])

ax_vid_a = fig.add_subplot(gs[:, 0])
ax_vid_b = fig.add_subplot(gs[:, 1])

ax_wav_a = fig.add_subplot(gs[0, 2])
ax_wav_b = fig.add_subplot(gs[1, 2], sharex=ax_wav_a)
ax_pnow = fig.add_subplot(gs[2, 2], sharex=ax_wav_a)
ax_pfut = fig.add_subplot(gs[3, 2], sharex=ax_wav_a)

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

for ax in (ax_wav_a, ax_wav_b, ax_pnow, ax_pfut, ax_wav_a_vad, ax_wav_b_vad):
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
            "-i",
            audio_a,
            "-i",
            audio_b,
            "-filter_complex",
            "amix=inputs=2:duration=longest",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            str(preview_sec),
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
