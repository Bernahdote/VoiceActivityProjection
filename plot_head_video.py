"""
Side-by-side: Video A | Video B, with head_encodings and alignment_head_rotation plots below.

Layout per column:
  [ Video ]
  [ head_encodings (3 dims) ]
  [ alignment_head_rotation (3 dims) ]

Usage:
    uv run plot_head_video.py
    uv run plot_head_video.py --start 10 --end 40 --window 5
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from tqdm import trange

SRC_FPS = 30.0

COL_W  = 960
VID_H  = 540
PLOT_H = 300   # height per plot panel (two panels per column)
OUT_W  = COL_W * 2
OUT_H  = VID_H + PLOT_H * 2

BASE_A = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A")
BASE_B = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A")

DIM_LABELS = ["Dim 0", "Dim 1", "Dim 2"]


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


def render_plot(ax, fig, t_win, data, t_now, title, ymin, ymax):
    ax.cla()
    colors = ["steelblue", "darkorange", "green"]
    for i in range(data.shape[1]):
        ax.plot(t_win, data[:, i], color=colors[i], label=DIM_LABELS[i], linewidth=1.2)
    ax.axvline(x=t_now, color="red", linewidth=1.2, linestyle="--")
    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    plot = fig_to_numpy(fig)
    return cv2.resize(plot, (COL_W, PLOT_H))


def fit_frame(frame):
    h, w = frame.shape[:2]
    scale = min(COL_W / w, VID_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((VID_H, COL_W, 3), dtype=np.uint8)
    y0 = (VID_H - new_h) // 2
    x0 = (COL_W - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  type=float, default=0.0)
    parser.add_argument("--end",    type=float, default=30.0)
    parser.add_argument("--window", type=float, default=5.0)
    args = parser.parse_args()

    # ── load features from source npz ────────────────────────────────────────
    za = np.load(BASE_A.with_suffix(".npz"), allow_pickle=False)
    zb = np.load(BASE_B.with_suffix(".npz"), allow_pickle=False)

    head_a  = za["movement:head_encodings"]
    head_b  = zb["movement:head_encodings"]
    align_a = za["movement:alignment_head_rotation"]
    align_b = zb["movement:alignment_head_rotation"]

    # ── open videos ──────────────────────────────────────────────────────────
    cap_a   = cv2.VideoCapture(str(BASE_A.with_suffix(".mp4")))
    cap_b   = cv2.VideoCapture(str(BASE_B.with_suffix(".mp4")))
    vid_fps = cap_a.get(cv2.CAP_PROP_FPS)

    start_frame = int(args.start * vid_fps)
    end_frame   = int(args.end   * vid_fps)
    half_window = int(args.window * SRC_FPS / 2)

    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # shared y-axis limits
    head_min  = min(head_a.min(),  head_b.min())  - 0.1
    head_max  = max(head_a.max(),  head_b.max())  + 0.1
    align_min = min(align_a.min(), align_b.min()) - 0.1
    align_max = max(align_a.max(), align_b.max()) + 0.1

    out_path = Path("head_video_tmp.mp4")
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (OUT_W, OUT_H))

    fig_ha, ax_ha = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)
    fig_hb, ax_hb = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)
    fig_aa, ax_aa = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)
    fig_ab, ax_ab = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)

    for frame_idx in trange(start_frame, end_frame, desc="Writing"):
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            break

        t_now    = frame_idx / vid_fps
        feat_idx = int(t_now * SRC_FPS)
        i0   = max(0, feat_idx - half_window)
        i1_a = min(len(head_a), feat_idx + half_window)
        i1_b = min(len(head_b), feat_idx + half_window)
        t_win_a = np.arange(i0, i1_a) / SRC_FPS
        t_win_b = np.arange(i0, i1_b) / SRC_FPS

        ph_a = render_plot(ax_ha, fig_ha, t_win_a, head_a[i0:i1_a],  t_now, "A: head_encodings",          head_min,  head_max)
        ph_b = render_plot(ax_hb, fig_hb, t_win_b, head_b[i0:i1_b],  t_now, "B: head_encodings",          head_min,  head_max)
        pa_a = render_plot(ax_aa, fig_aa, t_win_a, align_a[i0:i1_a], t_now, "A: alignment_head_rotation", align_min, align_max)
        pa_b = render_plot(ax_ab, fig_ab, t_win_b, align_b[i0:i1_b], t_now, "B: alignment_head_rotation", align_min, align_max)

        col_a = np.concatenate([fit_frame(frame_a), ph_a, pa_a], axis=0)
        col_b = np.concatenate([fit_frame(frame_b), ph_b, pa_b], axis=0)
        out.write(np.concatenate([col_a, col_b], axis=1))

    cap_a.release()
    cap_b.release()
    out.release()
    for f in [fig_ha, fig_hb, fig_aa, fig_ab]:
        plt.close(f)

    final = Path("head_video.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(out_path),
        "-ss", str(args.start), "-to", str(args.end),
        "-i", str(BASE_A.with_suffix(".wav")),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
        str(final),
    ], check=True)
    out_path.unlink()
    print(f"Saved → {final.resolve()}")


if __name__ == "__main__":
    main()
