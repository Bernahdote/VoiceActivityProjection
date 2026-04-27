"""
Video with AU26 (feature index 20, 0-based) value overlaid as a bar + number.
"""

import numpy as np
import cv2
import subprocess
from pathlib import Path
from tqdm import trange

# ── config ────────────────────────────────────────────────────────────────────
BASE      = Path("/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A")
AU26_IDX  = 20          # feature 21/24, 0-indexed
T_START   = 0.0         # seconds
T_END     = 40.0

# ── load FAU ─────────────────────────────────────────────────────────────────
z       = np.load(BASE.with_suffix(".npz"), allow_pickle=False)
au26    = z["movement:FAUValue"][:, AU26_IDX]   # (T_video,)
valid   = z["movement:is_valid"].astype(bool)
au26[~valid] = 0.0
src_fps = 30.0
au26_max = float(au26.max()) or 1.0

# ── open video ────────────────────────────────────────────────────────────────
cap     = cv2.VideoCapture(str(BASE.with_suffix(".mp4")))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ── output writer ─────────────────────────────────────────────────────────────
out_path = Path("fau_video_alignment.mp4")
fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
out      = cv2.VideoWriter(str(out_path), fourcc, vid_fps, (W, H))

start_frame = int(T_START * vid_fps)
end_frame   = int(T_END   * vid_fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

BAR_H      = 40          # height of bar gauge strip at bottom
BAR_MAX_W  = W - 40      # max bar width in pixels

for frame_idx in trange(start_frame, end_frame, desc="Writing"):
    ok, frame = cap.read()
    if not ok:
        break

    t_now   = frame_idx / vid_fps
    fau_idx = int(t_now * src_fps)
    fau_idx = min(fau_idx, len(au26) - 1)
    val     = float(au26[fau_idx])
    val_n   = val / au26_max          # 0 → 1

    # ── bar gauge at bottom ──────────────────────────────────────────────────
    bar_y   = H - BAR_H - 10
    bar_w   = int(val_n * BAR_MAX_W)
    color   = (0, int(255 * (1 - val_n)), int(255 * val_n))  # green→red

    cv2.rectangle(frame, (20, bar_y), (20 + BAR_MAX_W, bar_y + BAR_H),
                  (60, 60, 60), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (20, bar_y), (20 + bar_w, bar_y + BAR_H),
                      color, -1)
    cv2.rectangle(frame, (20, bar_y), (20 + BAR_MAX_W, bar_y + BAR_H),
                  (200, 200, 200), 1)

    # ── text labels ──────────────────────────────────────────────────────────
    cv2.putText(frame, f"AU26  {val:.3f}",
                (20, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{t_now:.2f}s",
                (W - 110, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out.write(frame)

cap.release()
out.release()

# ── mux original audio with ffmpeg ───────────────────────────────────────────
out_with_audio = Path("fau_video_alignment_audio.mp4")
subprocess.run([
    "ffmpeg", "-y",
    "-i", str(out_path),
    "-ss", str(T_START), "-to", str(T_END),
    "-i", str(BASE.with_suffix(".wav")),
    "-c:v", "copy",
    "-c:a", "aac", "-b:a", "192k",
    "-shortest",
    str(out_with_audio),
], check=True)

print(f"Saved (with audio) → {out_with_audio.resolve()}")
