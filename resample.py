import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


SR = 16000
PCM_CODEC = "pcm_s16le"
WORKERS = 8


def resample_file(path: Path) -> tuple[Path, bool, str]:
    tmp = path.with_name(f"{path.stem}.tmp.wav")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", str(SR), "-c:a", PCM_CODEC, str(tmp)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        tmp.replace(path)
        return path, True, ""
    except subprocess.CalledProcessError as e:
        if tmp.exists():
            tmp.unlink()
        return path, False, e.stderr.decode(errors="ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    root: Path = args.root
    assert root.is_dir(), f"Root not found: {root}"

    mp3s = list(root.rglob("*.mp3"))
    for f in mp3s:
        f.unlink()
    print(f"Removed {len(mp3s)} mp3 files.")

    wavs = list(root.rglob("*.wav"))
    print(f"Resampling {len(wavs)} wav files to {SR} Hz...")
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(resample_file, w) for w in wavs]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Resampling"):
            path, ok, err = f.result()
            if not ok:
                failures.append((path, err))
    if failures:
        print(f"\n{len(failures)} resampling failures:")
        for p, e in failures[:10]:
            print(f"  {p}: {e.splitlines()[-1] if e else ''}")


if __name__ == "__main__":
    main()
