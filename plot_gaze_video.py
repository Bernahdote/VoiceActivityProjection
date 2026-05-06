"""
Layout: single column (left person only)
  [ Video ]
  [ Gaze (yaw + pitch) ]

Usage:
    uv run plot_gaze_video.py
    uv run plot_gaze_video.py --start 10 --end 40 --window 5
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

GAZE_SLICE = slice(0, 2)
SRC_FPS    = 30.0

COL_W  = 960
VID_H  = 540
PLOT_H = 300
OUT_W  = COL_W
OUT_H  = VID_H + PLOT_H

BASE_A = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A")


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


def render_gaze_plot(ax, fig, t_win, yaw, pitch, t_now, gaze_min, gaze_max, label, half_win_sec):
    ax.cla()
    ax.plot(t_win, pitch, color="steelblue",  label="Pitch", linewidth=1.5)
    ax.plot(t_win, yaw,   color="darkorange", label="Yaw",   linewidth=1.5)
    ax.axvline(x=t_now, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlim(t_now - half_win_sec, t_now + half_win_sec)
    ax.set_ylim(gaze_min, gaze_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(label, fontsize=9)
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

    z_a    = np.load(BASE_A.with_suffix(".f.npz"), allow_pickle=False)
    gaze_a = np.degrees(z_a["features"][:, GAZE_SLICE])

    cap_a   = cv2.VideoCapture(str(BASE_A.with_suffix(".mp4")))
    vid_fps = cap_a.get(cv2.CAP_PROP_FPS)

    start_frame = int(args.start * vid_fps)
    end_frame   = int(args.end   * vid_fps)
    half_window = int(args.window * SRC_FPS / 2)

    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    gaze_min = float(gaze_a.min()) - 1.0
    gaze_max = float(gaze_a.max()) + 1.0

    out_path = Path("gaze_video_tmp.mp4")
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (OUT_W, OUT_H))

    fig_a, ax_a = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)

    for frame_idx in trange(start_frame, end_frame, desc="Writing"):
        ok_a, frame_a = cap_a.read()
        if not ok_a:
            break

        t_now    = frame_idx / vid_fps
        feat_idx = int(t_now * SRC_FPS)
        i0   = max(0, feat_idx - half_window)
        i1_a = min(len(gaze_a), feat_idx + half_window)
        t_win_a = np.arange(i0, i1_a) / SRC_FPS

        p_a = render_gaze_plot(ax_a, fig_a, t_win_a, gaze_a[i0:i1_a, 0], gaze_a[i0:i1_a, 1], t_now, gaze_min, gaze_max, "gaze_encodings", args.window / 2)

        out.write(np.concatenate([fit_frame(frame_a), p_a], axis=0))

    cap_a.release()
    out.release()
    plt.close(fig_a)

    final = Path("gaze_video.mp4")
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
