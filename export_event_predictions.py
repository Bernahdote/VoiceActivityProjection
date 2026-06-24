"""
Export event-level predictions for two models (M0 baseline, M5 all features)
to a single CSV for mixed-effects analysis in R.

Output columns:
  event_id, task, model_type, true_label, prediction, correct,
  dyad_id, speaker_id, conversation_id

Each event appears twice (once per model), with shared event_id so the two
rows are paired.

Usage:
  uv run python export_event_predictions.py \\
    --ckpt0 /path/to/baseline.ckpt \\
    --ckpt1 /path/to/m5.ckpt \\
    --test_pt0 /path/to/baseline_test_dir \\
    --test_pt1 /path/to/m5_test_dir \\
    --test_csv /path/to/sliding.csv \\
    [--video_dim1 364] \\
    --output event_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm

from vap.modules.lightning_module import VAPModule


EVENT_NAMES = ["hs", "ls", "sp"]
PID_PATTERN = re.compile(r"_(P\d+[A-Z]?)\.")


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def _load_module(ckpt_path: Path, cfg, device):
    module = VAPModule.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    if not hasattr(module.model, "video_dim"):
        module.model.video_dim = 0
    module.val_metric = instantiate(cfg.module.val_metric)
    return module.to(device).eval()


def _make_loader(test_pt: str, cfg, batch_size: int, num_workers: int):
    cfg = cfg.copy()
    cfg.datamodule.test_path = test_pt
    cfg.datamodule.batch_size = batch_size
    cfg.datamodule.num_workers = num_workers
    dm = instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup("test")
    return dm.test_dataloader()


def _extract_pid(path: str) -> str | None:
    m = PID_PATTERN.search(str(path))
    return m.group(1) if m else None


TASK_LABELS = {
    "HS": ("Hold", "Shift"),       # 0 = Hold, 1 = Shift
    "SL": ("Short", "Long"),       # 0 = Short, 1 = Long
    "PS": ("PreHold", "PreShift"), # 0 = PreHold, 1 = PreShift
}


EVENT_POSITIONS = (
    ("HS", "shift", 1),
    ("HS", "hold", 0),
    ("SL", "long", 1),
    ("SL", "short", 0),
    ("PS", "pred_shift", 1),
    ("PS", "pred_shift_neg", 0),
)


def _events_for_clip(events, b, T):
    """Return a flat list of (task, ev_pos, label_int, start, end, speaker)
    in a deterministic order, for clip index `b` in the batched event dict.
    """
    out = []
    for task, ev_pos, label_int in EVENT_POSITIONS:
        if ev_pos not in events:
            continue
        for start, end, speaker in events[ev_pos][b]:
            if end > T:
                end = T
            if end <= start:
                continue
            out.append((task, ev_pos, label_int, int(start), int(end), int(speaker)))
    return out


def _prediction_for_event(p_now, p_fut, b, task, ev_pos, start, end, speaker, threshold):
    """Compute the per-event aggregated prediction (0/1) and the channel of the
    user-specified speaker (HS: before silence, SL: incoming, PS: currently active).
    """
    if task == "HS":
        p_shift_of_speaker = p_now[b, start:end] if speaker == 0 else 1 - p_now[b, start:end]
        if ev_pos == "shift":
            p_event = p_shift_of_speaker
            speaker_channel = 1 - speaker
        else:
            p_event = 1 - p_shift_of_speaker
            speaker_channel = speaker
    elif task == "SL":
        p_event = p_fut[b, start:end] if speaker == 0 else 1 - p_fut[b, start:end]
        speaker_channel = speaker
    else:  # PS
        p_shift_of_speaker = p_fut[b, start:end] if speaker == 0 else 1 - p_fut[b, start:end]
        if ev_pos == "pred_shift":
            p_event = p_shift_of_speaker
        else:
            p_event = 1 - p_shift_of_speaker
        speaker_channel = 1 - speaker
    mean_p = float(p_event.mean())
    pred_int = 1 if mean_p >= threshold else 0
    return pred_int, speaker_channel


def _predictions_for_cached_events(
    module, loader, has_video: bool, per_clip_event_lists, threshold: float = 0.5,
):
    """Run inference and compute predictions for events that were already
    extracted on the M0 pass (so M0 and M5 use IDENTICAL event sets)."""
    per_clip_events: list[list[dict]] = []
    clip_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            batch = _to_device(batch, module.device)
            if has_video and getattr(module.model, "video_dim", 0) > 0:
                out = module.model(
                    batch["waveform"],
                    video_features_a=batch["video_features_a"],
                    video_features_b=batch["video_features_b"],
                )
            else:
                out = module.model(batch["waveform"])
            probs = module.model.objective.get_probs(out["logits"])
            p_now = probs["p_now"]
            p_fut = probs["p_future"]
            bsz = batch["waveform"].shape[0]
            for b in range(bsz):
                clip_events = []
                for intra_idx, (task, ev_pos, label_int, start, end, speaker) in enumerate(
                    per_clip_event_lists[clip_idx]
                ):
                    pred_int, speaker_channel = _prediction_for_event(
                        p_now, p_fut, b, task, ev_pos, start, end, speaker, threshold
                    )
                    clip_events.append({
                        "intra_idx": intra_idx,
                        "task": task,
                        "true_label_int": label_int,
                        "prediction_int": pred_int,
                        "speaker_channel": speaker_channel,
                    })
                per_clip_events.append(clip_events)
                clip_idx += 1
    return per_clip_events


def _per_clip_event_predictions(
    module, loader, has_video: bool, threshold: float = 0.5, return_events: bool = False,
):
    """Return list[list[event dict]]: outer index = clip in loader order.

    Each event dict has:
        intra_idx (int)   - deterministic intra-clip event index for join
        task              - 'HS' | 'SL' | 'PS'
        true_label_int    - 0 or 1
        prediction_int    - 0 or 1 (after threshold)
        speaker_channel   - 0 or 1, channel of the speaker the user cares about
                            (HS: speaker active before silence,
                             SL: incoming speaker,
                             PS: currently active speaker)
    """
    metric = module.val_metric
    per_clip_events: list[list[dict]] = []
    per_clip_event_lists: list[list[tuple]] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference"):
            batch = _to_device(batch, module.device)
            if has_video and getattr(module.model, "video_dim", 0) > 0:
                out = module.model(
                    batch["waveform"],
                    video_features_a=batch["video_features_a"],
                    video_features_b=batch["video_features_b"],
                )
            else:
                out = module.model(batch["waveform"])

            probs = module.model.objective.get_probs(out["logits"])
            events = metric.event_extractor(batch["vad"])

            bsz = batch["waveform"].shape[0]
            T = batch["vad"].shape[1]
            p_now = probs["p_now"]
            p_fut = probs["p_future"]

            for b in range(bsz):
                clip_events: list[dict] = []
                # Build deterministic event list for this clip and store it
                ev_list = _events_for_clip(events, b, T)
                per_clip_event_lists.append(ev_list)
                for intra_idx, (task, ev_pos, label_int, start, end, speaker) in enumerate(ev_list):
                    pred_int, speaker_channel = _prediction_for_event(
                        p_now, p_fut, b, task, ev_pos, start, end, speaker, threshold
                    )
                    clip_events.append({
                        "intra_idx": intra_idx,
                        "task": task,
                        "true_label_int": label_int,
                        "prediction_int": pred_int,
                        "speaker_channel": speaker_channel,
                    })
                per_clip_events.append(clip_events)
    if return_events:
        return per_clip_events, per_clip_event_lists
    return per_clip_events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt0", required=True)
    parser.add_argument("--ckpt1", required=True)
    parser.add_argument("--test_pt0", required=True)
    parser.add_argument("--test_pt1", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--video_dim0", type=int, default=None)
    parser.add_argument("--video_dim1", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_config", default="vap/conf/stereo_home_dev.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import pandas as pd
    df = pd.read_csv(args.test_csv)
    # extract speaker IDs and conversation IDs for each test row
    df["pid_a"] = df["audio_path_a"].astype(str).apply(_extract_pid)
    df["pid_b"] = df["audio_path_b"].astype(str).apply(_extract_pid)
    df["dyad_id"] = df.apply(
        lambda r: "_".join(sorted([r["pid_a"] or "?", r["pid_b"] or "?"])), axis=1
    )
    df["conversation_id"] = df["session"].astype(str)

    # ---- Model 0: get probs, extract events, save events + predictions ----
    cfg0 = OmegaConf.load(args.model_config)
    if args.video_dim0 is not None:
        cfg0.module.model.video_dim = args.video_dim0
    loader0 = _make_loader(args.test_pt0, cfg0, args.batch_size, args.num_workers)
    m0 = _load_module(Path(args.ckpt0), cfg0, device)
    print("Inference for model M0...")
    # IMPORTANT: seed before event extraction so pred_shift_neg negatives are
    # reproducible. We then reuse the SAME extracted events for M5 to guarantee
    # equal event counts and a paired event_id.
    import random
    random.seed(0)
    torch.manual_seed(0)
    has_video0 = args.video_dim0 is not None and args.video_dim0 > 0
    events_m0, per_clip_event_lists = _per_clip_event_predictions(
        m0, loader0, has_video0, threshold=args.threshold, return_events=True,
    )
    del m0

    # ---- Model 1: rerun the same loader, REUSE the same events ----
    cfg1 = OmegaConf.load(args.model_config)
    if args.video_dim1 is not None:
        cfg1.module.model.video_dim = args.video_dim1
    loader1 = _make_loader(args.test_pt1, cfg1, args.batch_size, args.num_workers)
    m1 = _load_module(Path(args.ckpt1), cfg1, device)
    print("Inference for model M5 (using cached events from M0 run)...")
    has_video1 = args.video_dim1 is not None and args.video_dim1 > 0
    events_m1 = _predictions_for_cached_events(
        m1, loader1, has_video1, per_clip_event_lists, threshold=args.threshold,
    )
    del m1

    assert len(events_m0) == len(events_m1) == len(df), \
        f"Mismatched clip count: M0={len(events_m0)}, M5={len(events_m1)}, csv={len(df)}"

    # ---- Write CSV ----
    print(f"\nWriting {args.output}...")
    n_rows = 0
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "event_id", "task", "model_type", "true_label",
            "prediction", "correct", "dyad_id", "speaker_id", "conversation_id",
        ])
        for clip_idx in range(len(df)):
            row = df.iloc[clip_idx]
            dyad = row["dyad_id"]
            conv = row["conversation_id"]
            pids = (row["pid_a"], row["pid_b"])
            clip_m0 = events_m0[clip_idx]
            clip_m1 = events_m1[clip_idx]
            assert len(clip_m0) == len(clip_m1), \
                f"clip {clip_idx}: event count differs M0={len(clip_m0)} vs M5={len(clip_m1)}"
            for e0, e1 in zip(clip_m0, clip_m1):
                assert (e0["task"], e0["true_label_int"], e0["speaker_channel"]) == \
                       (e1["task"], e1["true_label_int"], e1["speaker_channel"]), \
                    f"event metadata mismatch at clip {clip_idx} idx {e0['intra_idx']}"
                event_id = f"{conv}_{e0['intra_idx']}"
                speaker_id = pids[e0["speaker_channel"]] or "?"
                neg_label, pos_label = TASK_LABELS[e0["task"]]
                true_label = pos_label if e0["true_label_int"] == 1 else neg_label
                m0_pred = pos_label if e0["prediction_int"] == 1 else neg_label
                m1_pred = pos_label if e1["prediction_int"] == 1 else neg_label
                w.writerow([event_id, e0["task"], "M0", true_label,
                            m0_pred, int(e0["prediction_int"] == e0["true_label_int"]),
                            dyad, speaker_id, conv])
                w.writerow([event_id, e1["task"], "M5", true_label,
                            m1_pred, int(e1["prediction_int"] == e1["true_label_int"]),
                            dyad, speaker_id, conv])
                n_rows += 2
    print(f"Done. Wrote {n_rows} rows to {args.output}.")


if __name__ == "__main__":
    main()
