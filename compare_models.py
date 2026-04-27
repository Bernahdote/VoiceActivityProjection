"""
Comparison: Baseline vs FAU model.
Two videos + waveforms + predicted VAD + p_now + p_future + per-frame loss diff.
Visualization style mirrors running-visual.correct.py.
"""

import json
import math
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch

from vap.data.create_sliding_window_dset import sliding_window
from vap.data.datamodule import VAPDataModule
from vap.modules.lightning_module import VAPModule
from vap.utils.audio import load_waveform

# ── config ────────────────────────────────────────────────────────────────────
CKPT_BASELINE = "/Users/willemberner/Desktop/Exjobb/checkpoints/baseline/epoch=6-step=22505.ckpt"
CKPT_FAU      = "/Users/willemberner/Desktop/Exjobb/checkpoints/FAU/epoch=11-step=38580.ckpt"

AUDIO_A = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0038/V00_S2020_I00000686_P1275A.wav"
AUDIO_B = "/Users/willemberner/datasets/seamless_interaction/improvised/dev/0000/0032/V00_S2020_I00000686_P1276A.wav"

WINDOW_DURATION = 20.0
WINDOW_OVERLAP  = 5.0
WINDOW_HORIZON  = 2.0
MAX_WINDOWS     = 6
PREVIEW_SECONDS = 60.0
OUT_FPS         = 25
VIDEO_WIDTH     = 240
OUT_PATH        = "compare_baseline_fau.mp4"
THRESHOLD       = 0.5

COL_BASE  = "#4878CF"   # blue  — Baseline model
COL_FAU   = "#6ACC65"   # green — FAU model
COL_WAV_A = "royalblue"
COL_WAV_B = "orange"
SMOOTH_W  = 15


# ── helpers ───────────────────────────────────────────────────────────────────
def run_ffmpeg(args):
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_datamodule(csv_path: Path, video_feature_groups: list) -> VAPDataModule:
    dm = VAPDataModule(
        test_path=str(csv_path), horizon=WINDOW_HORIZON,
        sample_rate=16000, frame_hz=50,
        mono=False, batch_size=1, num_workers=0,
        pin_memory=False, prefetch_factor=None,
        video_feature_groups=video_feature_groups,
    )
    dm.prepare_data()
    dm.setup("test")
    return dm


def build_sample_csv(tmp_dir: Path):
    vad_a = [[x["start"], x["end"]] for x in
             json.load(open(AUDIO_A.replace(".wav", ".json"))).get("metadata:vad", [])]
    vad_b = [[x["start"], x["end"]] for x in
             json.load(open(AUDIO_B.replace(".wav", ".json"))).get("metadata:vad", [])]
    samples = sliding_window(
        vad_list=[vad_a, vad_b],
        audio_path_a=AUDIO_A, audio_path_b=AUDIO_B,
        duration=WINDOW_DURATION, overlap=WINDOW_OVERLAP, horizon=WINDOW_HORIZON,
    )[:MAX_WINDOWS]
    payload = {k: [] for k in ("session","audio_path_a","audio_path_b","start","end","vad_list","dataset")}
    for s in samples:
        for k in ("session","audio_path_a","audio_path_b","start","end"):
            payload[k].append(s[k])
        payload["vad_list"].append(json.dumps(s["vad_list"]))
        payload["dataset"].append("custom")
    df = pd.DataFrame(payload)
    csv_path = tmp_dir / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def run_inference(model, dm, device):
    """Mirrors evaluate_model-batch.py.
    Returns (loss, p_now, p_future, vad_pred)
      loss/p_now/p_future : (T,)   per-frame
      vad_pred            : (T, 2) model predicted VAD (sigmoid)
    """
    step_frames = int(round((WINDOW_DURATION - WINDOW_OVERLAP) * model.frame_hz))
    loss_agg = p_now_agg = p_fut_agg = vpred_agg = None

    with torch.inference_mode():
        for batch in dm.test_dataloader():
            waveform     = batch["waveform"].to(device)
            vad          = batch["vad"].to(device)
            video_feat_a = batch["video_features_a"].to(device)
            video_feat_b = batch["video_features_b"].to(device)

            out = model(waveform,
                        video_features_a=video_feat_a,
                        video_features_b=video_feat_b)

            labels   = model.extract_labels(vad)
            loss     = model.objective.loss_vap(out["logits"], labels, reduction="none")[0]
            vad_pred = out["vad"][0].sigmoid().cpu()   # (T, 2)

            p_agg = model.objective.get_probs(out["logits"])
            p_now = p_agg["p_now"][0]
            p_fut = p_agg["p_future"][0]

            if loss_agg is None:
                loss_agg  = loss
                p_now_agg = p_now
                p_fut_agg = p_fut
                vpred_agg = vad_pred
            else:
                loss_agg  = torch.cat([loss_agg,  loss[-step_frames:]])
                p_now_agg = torch.cat([p_now_agg, p_now[-step_frames:]])
                p_fut_agg = torch.cat([p_fut_agg, p_fut[-step_frames:]])
                vpred_agg = torch.cat([vpred_agg, vad_pred[-step_frames:]])

    return loss_agg, p_now_agg, p_fut_agg, vpred_agg


def smooth(y: np.ndarray, w: int = SMOOTH_W) -> np.ndarray:
    return np.convolve(y, np.ones(w) / w, mode="same")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading baseline model...")
    mod_base   = VAPModule.load_from_checkpoint(CKPT_BASELINE, map_location=device, weights_only=False)
    model_base = mod_base.model.to(device).eval()
    if not hasattr(model_base, "video_dim"):
        model_base.video_dim = 0
    # Baseline was trained with nn.Identity() feature_projection (encoder.dim == transformer.dim == 256).
    # Current branch replaced it with an MLP — reset it so the loaded transformer weights see
    # the same representation they were trained with.
    model_base.feature_projection = torch.nn.Identity()

    print("Loading FAU model...")
    mod_fau   = VAPModule.load_from_checkpoint(CKPT_FAU, map_location=device, weights_only=False)
    model_fau = mod_fau.model.to(device).eval()

    frame_hz = model_fau.frame_hz
    sr       = model_fau.sample_rate

    print("Building dataset...")
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path, samples_df = build_sample_csv(Path(tmpdir))

        print("Running baseline inference...")
        dm_base = build_datamodule(csv_path, video_feature_groups=["fauv"])
        loss_base, p_now_base, p_fut_base, vad_pred_base = run_inference(model_base, dm_base, device)
        print(f"  Baseline  — loss mean: {loss_base.mean():.4f}  min: {loss_base.min():.4f}  max: {loss_base.max():.4f}")

        print("Running FAU inference...")
        dm_fau = build_datamodule(csv_path, video_feature_groups=["fauv"])
        loss_fau, p_now_fau, p_fut_fau, vad_pred_fau = run_inference(model_fau, dm_fau, device)
        print(f"  FAU       — loss mean: {loss_fau.mean():.4f}  min: {loss_fau.min():.4f}  max: {loss_fau.max():.4f}")

    segment_start = float(samples_df["start"].min())
    segment_end   = float(samples_df["end"].max())
    preview_sec   = min(PREVIEW_SECONDS, segment_end - segment_start)

    n_frames = min(len(loss_base), len(loss_fau), len(p_now_base), len(p_now_fau))
    x = np.arange(n_frames) / frame_hz

    loss_base_s  = smooth(loss_base[:n_frames].numpy())
    loss_fau_s   = smooth(loss_fau[:n_frames].numpy())
    p_now_base_s = smooth(p_now_base[:n_frames].numpy())
    p_now_fau_s  = smooth(p_now_fau[:n_frames].numpy())
    p_fut_base_s = smooth(p_fut_base[:n_frames].numpy())
    p_fut_fau_s  = smooth(p_fut_fau[:n_frames].numpy())
    diff_s       = loss_base_s - loss_fau_s

    vad_base = vad_pred_base[:n_frames].numpy()   # (T, 2)
    vad_fau  = vad_pred_fau[:n_frames].numpy()    # (T, 2)

    # load waveforms from file — same as running-visual.correct.py
    w1, _ = load_waveform(AUDIO_A, sample_rate=sr, mono=True,
                          start_time=segment_start, end_time=segment_end)
    w2, _ = load_waveform(AUDIO_B, sample_rate=sr, mono=True,
                          start_time=segment_start, end_time=segment_end)
    n_samp  = min(w1.shape[-1], w2.shape[-1])
    wav_a   = w1[0, :n_samp].numpy()
    wav_b   = w2[0, :n_samp].numpy()
    t_audio = np.arange(n_samp) / sr

    # ── figure layout ─────────────────────────────────────────────────────────
    #  col 0: video A | col 1: video B | col 2: 5 stacked signal panels
    fig = plt.figure(figsize=(24, 14))
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 2.4], wspace=0.08)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.95, bottom=0.05, hspace=0.35)

    ax_vid_a = fig.add_subplot(gs[0]); ax_vid_a.axis("off"); ax_vid_a.set_title("Speaker A", fontsize=12)
    ax_vid_b = fig.add_subplot(gs[1]); ax_vid_b.axis("off"); ax_vid_b.set_title("Speaker B", fontsize=12)

    gs_right = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[2], hspace=0.50)
    ax_wav_a = fig.add_subplot(gs_right[0])
    ax_wav_b = fig.add_subplot(gs_right[1], sharex=ax_wav_a)
    ax_pnow  = fig.add_subplot(gs_right[2], sharex=ax_wav_a)
    ax_pfut  = fig.add_subplot(gs_right[3], sharex=ax_wav_a)
    ax_loss  = fig.add_subplot(gs_right[4], sharex=ax_wav_a)

    # ── waveform A + predicted VAD A ──────────────────────────────────────────
    ax_wav_a.plot(t_audio, wav_a, color=COL_WAV_A, alpha=0.9, lw=0.7)
    wav_a_lim = max(0.05, float(np.max(np.abs(wav_a))) * 1.15)
    ax_wav_a.set_ylim(-wav_a_lim, wav_a_lim)
    ax_wav_a.set_title("Waveform A", fontsize=10)
    ax_wav_a.margins(x=0)

    ax_vad_a = ax_wav_a.twinx()
    ax_vad_a.fill_between(x, 0, vad_base[:, 0], color=COL_BASE, alpha=0.25, label="Baseline")
    ax_vad_a.fill_between(x, 0, vad_fau[:, 0],  color=COL_FAU,  alpha=0.25, label="FAU")
    ax_vad_a.set_ylim(-0.05, 1.05)
    ax_vad_a.set_ylabel("VAD prob", fontsize=8)
    ax_vad_a.legend(fontsize=8, loc="upper right")

    # ── waveform B + predicted VAD B ──────────────────────────────────────────
    ax_wav_b.plot(t_audio, wav_b, color=COL_WAV_B, alpha=0.9, lw=0.7)
    wav_b_lim = max(0.05, float(np.max(np.abs(wav_b))) * 1.15)
    ax_wav_b.set_ylim(-wav_b_lim, wav_b_lim)
    ax_wav_b.set_title("Waveform B", fontsize=10)
    ax_wav_b.margins(x=0)

    ax_vad_b = ax_wav_b.twinx()
    ax_vad_b.fill_between(x, 0, vad_base[:, 1], color=COL_BASE, alpha=0.25, label="Baseline")
    ax_vad_b.fill_between(x, 0, vad_fau[:, 1],  color=COL_FAU,  alpha=0.25, label="FAU")
    ax_vad_b.set_ylim(-0.05, 1.05)
    ax_vad_b.set_ylabel("VAD prob", fontsize=8)
    ax_vad_b.legend(fontsize=8, loc="upper right")

    # ── p_now ─────────────────────────────────────────────────────────────────
    ax_pnow.axhline(THRESHOLD, color="k", ls="--", lw=1)
    # shading: blue family = A predicted, orange/gold family = B predicted
    # darker shade = Baseline, lighter shade = FAU
    ax_pnow.fill_between(x, p_now_base_s, THRESHOLD, where=(p_now_base_s >= THRESHOLD),
                         color="royalblue",      alpha=0.25, step="post")
    ax_pnow.fill_between(x, p_now_base_s, THRESHOLD, where=(p_now_base_s <  THRESHOLD),
                         color="orange",          alpha=0.25, step="post")
    ax_pnow.fill_between(x, p_now_fau_s,  THRESHOLD, where=(p_now_fau_s  >= THRESHOLD),
                         color="cornflowerblue",  alpha=0.25, step="post")
    ax_pnow.fill_between(x, p_now_fau_s,  THRESHOLD, where=(p_now_fau_s  <  THRESHOLD),
                         color="gold",            alpha=0.25, step="post")
    ax_pnow.step(x, p_now_base_s, where="post", color=COL_BASE, lw=1.4, label="Baseline")
    ax_pnow.step(x, p_now_fau_s,  where="post", color=COL_FAU,  lw=1.4, label="FAU")
    ax_pnow.set_ylim(-0.05, 1.05)
    ax_pnow.set_title("p_now  (blue=A predicted, orange=B predicted)", fontsize=10)
    ax_pnow.set_ylabel("P(A next — now)", fontsize=9)
    ax_pnow.legend(fontsize=8, loc="upper right")
    ax_pnow.grid(alpha=0.3)
    ax_pnow.margins(x=0)

    # ── p_future ──────────────────────────────────────────────────────────────
    ax_pfut.axhline(THRESHOLD, color="k", ls="--", lw=1)
    ax_pfut.fill_between(x, p_fut_base_s, THRESHOLD, where=(p_fut_base_s >= THRESHOLD),
                         color="royalblue",      alpha=0.25, step="post")
    ax_pfut.fill_between(x, p_fut_base_s, THRESHOLD, where=(p_fut_base_s <  THRESHOLD),
                         color="orange",          alpha=0.25, step="post")
    ax_pfut.fill_between(x, p_fut_fau_s,  THRESHOLD, where=(p_fut_fau_s  >= THRESHOLD),
                         color="cornflowerblue",  alpha=0.25, step="post")
    ax_pfut.fill_between(x, p_fut_fau_s,  THRESHOLD, where=(p_fut_fau_s  <  THRESHOLD),
                         color="gold",            alpha=0.25, step="post")
    ax_pfut.step(x, p_fut_base_s, where="post", color=COL_BASE, lw=1.4, label="Baseline")
    ax_pfut.step(x, p_fut_fau_s,  where="post", color=COL_FAU,  lw=1.4, label="FAU")
    ax_pfut.set_ylim(-0.05, 1.05)
    ax_pfut.set_title("p_future  (blue=A predicted, orange=B predicted)", fontsize=10)
    ax_pfut.set_ylabel("P(A next — future)", fontsize=9)
    ax_pfut.legend(fontsize=8, loc="upper right")
    ax_pfut.grid(alpha=0.3)
    ax_pfut.margins(x=0)

    # ── loss difference ───────────────────────────────────────────────────────
    ax_loss.axhline(0, color="gray", lw=1.0, zorder=1)
    ax_loss.fill_between(x, diff_s, 0, where=(diff_s >= 0), color=COL_FAU,  alpha=0.45, label="FAU better")
    ax_loss.fill_between(x, diff_s, 0, where=(diff_s <  0), color=COL_BASE, alpha=0.45, label="Baseline better")
    ax_loss.plot(x, diff_s, color="black", lw=1.2, zorder=2)
    y_lim = max(abs(diff_s).max() * 1.15, 0.5)
    ax_loss.set_ylim(-y_lim, y_lim)
    ax_loss.set_title("Per-frame loss difference  (baseline − FAU)", fontsize=10)
    ax_loss.set_xlabel("Time (s)", fontsize=10)
    ax_loss.set_ylabel("Loss diff", fontsize=9)
    ax_loss.legend(fontsize=8, loc="upper right")
    ax_loss.grid(alpha=0.3)
    ax_loss.margins(x=0)

    # shared xlim init
    for ax in (ax_wav_a, ax_wav_b, ax_pnow, ax_pfut, ax_loss):
        ax.set_xlim(0, float(t_audio[-1]))

    # cursors on main axes (twinx follows automatically via sharex)
    cursor_lines = [
        ax_wav_a.axvline(0, color="red", lw=1.5),
        ax_wav_b.axvline(0, color="red", lw=1.5),
        ax_pnow.axvline(0,  color="red", lw=1.5),
        ax_pfut.axvline(0,  color="red", lw=1.5),
        ax_loss.axvline(0,  color="red", lw=1.5),
    ]
    scroll_axes = [ax_wav_a, ax_wav_b, ax_pnow, ax_pfut, ax_loss]

    # ── render ────────────────────────────────────────────────────────────────
    video_a = str(Path(AUDIO_A).with_suffix(".mp4"))
    video_b = str(Path(AUDIO_B).with_suffix(".mp4"))
    total_frames = int(math.ceil(preview_sec * OUT_FPS))
    half_win     = 20.0 / 2.0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vframes_a  = tmpdir / "va";     vframes_a.mkdir()
        vframes_b  = tmpdir / "vb";     vframes_b.mkdir()
        frames_dir = tmpdir / "frames"; frames_dir.mkdir()

        print("Extracting video frames A...")
        run_ffmpeg(["ffmpeg", "-y", "-ss", str(segment_start), "-i", video_a,
                    "-vf", f"fps={OUT_FPS},scale={VIDEO_WIDTH}:-2",
                    "-q:v", "4", "-t", str(preview_sec),
                    str(vframes_a / "frame_%06d.jpg")])

        print("Extracting video frames B...")
        run_ffmpeg(["ffmpeg", "-y", "-ss", str(segment_start), "-i", video_b,
                    "-vf", f"fps={OUT_FPS},scale={VIDEO_WIDTH}:-2",
                    "-q:v", "4", "-t", str(preview_sec),
                    str(vframes_b / "frame_%06d.jpg")])

        fa_paths = sorted(vframes_a.glob("frame_*.jpg"))
        fb_paths = sorted(vframes_b.glob("frame_*.jpg"))
        im_a = ax_vid_a.imshow(plt.imread(fa_paths[0]))
        im_b = ax_vid_b.imshow(plt.imread(fb_paths[0]))

        print(f"Rendering {total_frames} frames...")
        for i in range(total_frames):
            t = i / OUT_FPS
            for ax in scroll_axes:
                ax.set_xlim(t - half_win, t + half_win)
            for line in cursor_lines:
                line.set_xdata([t, t])
            im_a.set_data(plt.imread(fa_paths[min(i, len(fa_paths) - 1)]))
            im_b.set_data(plt.imread(fb_paths[min(i, len(fb_paths) - 1)]))
            fig.savefig(frames_dir / f"frame_{i:06d}.png", dpi=80)
            if i % 100 == 0:
                print(f"  {i}/{total_frames}")

        print("Mixing audio...")
        mixed = tmpdir / "mixed.m4a"
        run_ffmpeg(["ffmpeg", "-y",
                    "-ss", str(segment_start), "-t", str(preview_sec), "-i", AUDIO_A,
                    "-ss", str(segment_start), "-t", str(preview_sec), "-i", AUDIO_B,
                    "-filter_complex", "amix=inputs=2:duration=longest",
                    "-c:a", "aac", "-b:a", "192k", str(mixed)])

        print("Muxing final video...")
        run_ffmpeg(["ffmpeg", "-y",
                    "-r", str(OUT_FPS), "-start_number", "0",
                    "-i", str(frames_dir / "frame_%06d.png"),
                    "-i", str(mixed),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest",
                    OUT_PATH])

    print(f"\nDone → {OUT_PATH}")
