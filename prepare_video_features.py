from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from tqdm import tqdm



#ROOT = Path("/mnt/sda/willem/datasets/seamless_interaction") #Linux
ROOT = Path("/Users/willemberner/datasets/seamless_interaction") #Mac



FEATURE_KEYS = [
    "movement:gaze_encodings",
    "movement:head_encodings",
    "movement:expression",
    "movement:alignment_head_rotation",
    "movement:FAUToken",
    "smplh:body_pose",
    "smplh:left_hand_pose",
    "smplh:right_hand_pose",
]
OUTPUT_TAIL = ".f.npz"


def to_2d(a: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        return a.reshape(-1, 1)
    if a.ndim == 0:
        return a.reshape(1, 1)
    d = int(np.prod(a.shape[1:], dtype=np.int64))
    return a.reshape(a.shape[0], d)


def build_fused_features(z: np.lib.npyio.NpzFile) -> np.ndarray:
    arrs = [to_2d(z[k]) for k in FEATURE_KEYS]
    t = min(
        [a.shape[0] for a in arrs]
        + [z["movement:is_valid"].shape[0], z["smplh:is_valid"].shape[0]]
    )
    arrs = [a[:t] for a in arrs]

    valid = z["movement:is_valid"][:t].reshape(-1).astype(bool)
    valid &= z["smplh:is_valid"][:t].reshape(-1).astype(bool)
    fused = np.concatenate(arrs, axis=-1)
    if not valid.all():
        fused[~valid] = 0.0
    return fused.astype(np.float32)


def output_path(inp: Path, input_root: Path, output_root: Path) -> Path:
    rel = inp.relative_to(input_root)
    out = output_root / rel
    return out.with_name(f"{out.stem}{OUTPUT_TAIL}")


def is_input_npz(path: Path) -> bool:
    return path.suffix == ".npz" and not path.name.endswith(OUTPUT_TAIL)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they exist.",
    )
    args = parser.parse_args()

    input_root = ROOT.resolve()
    output_root = input_root
    files = sorted(p for p in input_root.rglob("*.npz") if is_input_npz(p))
    print(f"Found {len(files)} source files under {input_root}")

    n_ok = 0
    n_skip = 0
    n_fail = 0

    for src in tqdm(files, desc="Preparing fused video features"):
        dst = output_path(src, input_root, output_root)
        if dst.exists() and not args.overwrite:
            n_skip += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            z = np.load(src, allow_pickle=False)
            fused = build_fused_features(z)
            np.savez_compressed(dst, features=fused)
            n_ok += 1
        except Exception as exc:
            n_fail += 1
            print(f"[FAIL] {src}: {exc}")

    print(f"Done. created={n_ok} skipped={n_skip} failed={n_fail}")


if __name__ == "__main__":
    main()
