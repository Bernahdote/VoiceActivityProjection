"""
Side-by-side: Video A | Video B | Gaze plot A | Gaze plot B

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

# Output layout: two columns (A and B), video on top, plot below
COL_W  = 960   # width of each column
VID_H  = 720   # height of the video strip
PLOT_H = 360   # height of the gaze plot
OUT_W  = COL_W * 2
OUT_H  = VID_H + PLOT_H


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """Render matplotlib figure to BGR numpy array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    return cv2.cvtColor(buf[:, :, :3], cv2.COLOR_RGB2BGR)


def render_gaze_plot(ax, fig, t_win, yaw, pitch, t_now, gaze_min, gaze_max, label):
    ax.cla()
    ax.plot(t_win, yaw,   color="steelblue",  label="Yaw (left-right)", linewidth=1.5)
    ax.plot(t_win, pitch, color="darkorange", label="Pitch (up-down)",  linewidth=1.5)
    ax.axvline(x=t_now, color="red", linewidth=1.5, linestyle="--")
    ax.set_xlim(t_win[0], t_win[-1])
    ax.set_ylim(gaze_min, gaze_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Gaze (°)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{label}  t={t_now:.2f}s", fontsize=10)
    fig.tight_layout()
    return fig_to_numpy(fig)



def main():
    BASE_A = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A")
    BASE_B = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A")

    parser = argparse.ArgumentParser()
    parser.add_argument("--start",  type=float, default=0.0,  help="Start time (s)")
    parser.add_argument("--end",    type=float, default=30.0, help="End time (s)")
    parser.add_argument("--window", type=float, default=5.0,  help="Rolling window size (s)")
    args = parser.parse_args()

    # ── load gaze ────────────────────────────────────────────────────────────
    z_a    = np.load(BASE_A.with_suffix(".f.npz"), allow_pickle=False)
    z_b    = np.load(BASE_B.with_suffix(".f.npz"), allow_pickle=False)
    gaze_a = np.degrees(z_a["features"][:, GAZE_SLICE])
    gaze_b = np.degrees(z_b["features"][:, GAZE_SLICE])

    # ── open videos ──────────────────────────────────────────────────────────
    cap_a   = cv2.VideoCapture(str(BASE_A.with_suffix(".mp4")))
    cap_b   = cv2.VideoCapture(str(BASE_B.with_suffix(".mp4")))
    vid_fps = cap_a.get(cv2.CAP_PROP_FPS)
    W       = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    H       = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))

    t_end       = args.end if args.end is not None else gaze_a.shape[0] / SRC_FPS
    start_frame = int(args.start * vid_fps)
    end_frame   = int(t_end * vid_fps)
    half_window = int(args.window * SRC_FPS / 2)

    cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    gaze_min = min(gaze_a.min(), gaze_b.min()) - 1.0
    gaze_max = max(gaze_a.max(), gaze_b.max()) + 1.0

    # ── output writer ────────────────────────────────────────────────────────
    out_path = Path("gaze_video_tmp.mp4")
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), vid_fps, (OUT_W, OUT_H))

    fig_a, ax_a = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)
    fig_b, ax_b = plt.subplots(figsize=(COL_W / 100, PLOT_H / 100), dpi=100)

    for frame_idx in trange(start_frame, end_frame, desc="Writing"):
        ok_a, frame_a = cap_a.read()
        ok_b, frame_b = cap_b.read()
        if not ok_a or not ok_b:
            break

        t_now    = frame_idx / vid_fps
        feat_idx = int(t_now * SRC_FPS)
        i0   = max(0, feat_idx - half_window)
        i1_a = min(len(gaze_a), feat_idx + half_window)
        i1_b = min(len(gaze_b), feat_idx + half_window)
        t_win_a = np.arange(i0, i1_a) / SRC_FPS
        t_win_b = np.arange(i0, i1_b) / SRC_FPS

        plot_a = render_gaze_plot(ax_a, fig_a, t_win_a, gaze_a[i0:i1_a, 0], gaze_a[i0:i1_a, 1], t_now, gaze_min, gaze_max, "Speaker A")
        plot_b = render_gaze_plot(ax_b, fig_b, t_win_b, gaze_b[i0:i1_b, 0], gaze_b[i0:i1_b, 1], t_now, gaze_min, gaze_max, "Speaker B")

        # Scale videos keeping aspect ratio, center in VID_H x COL_W canvas
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

        vid_a = fit_frame(frame_a)
        vid_b = fit_frame(frame_b)

        # Scale plots to COL_W wide, PLOT_H tall
        plt_a = cv2.resize(plot_a, (COL_W, PLOT_H))
        plt_b = cv2.resize(plot_b, (COL_W, PLOT_H))

        # Stack video on top of plot per column, then side by side
        col_a = np.concatenate([vid_a, plt_a], axis=0)
        col_b = np.concatenate([vid_b, plt_b], axis=0)
        combined = np.concatenate([col_a, col_b], axis=1)
        out.write(combined)

    cap_a.release()
    cap_b.release()
    out.release()
    plt.close(fig_a)
    plt.close(fig_b)

    # ── mux audio ────────────────────────────────────────────────────────────
    final = Path("gaze_video.mp4")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(out_path),
        "-ss", str(args.start), "-to", str(t_end),
        "-i", str(BASE_A.with_suffix(".wav")),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(final),
    ], check=True)
    out_path.unlink()

    print(f"Saved → {final.resolve()}")


if __name__ == "__main__":
    main()
