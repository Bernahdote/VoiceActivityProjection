import json
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf


TARGET_SR = 16000


def read_vad_list(path: Path) -> List[List[float]]:
    """
    Extracts metadata:vad from a seamless_interaction json file and converts it
    to VAP VAD-list format: [[start, end], ...]
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    vad = data.get("metadata:vad", [])
    return [[float(v["start"]), float(v["end"])] for v in vad]


def resample_wav(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Loads a mono wav and resamples to target_sr. Returns float32 audio.
    """
    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono wav at {path}, got shape {audio.shape}")
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def find_pairs(conv_dir: Path) -> List[Tuple[Path, Path]]:
    """
    Returns list of (wav_path, json_path) pairs for a conversation folder.
    Matches by stem.
    """
    wavs = {p.stem: p for p in conv_dir.glob("*.wav")}
    jsons = {p.stem: p for p in conv_dir.glob("*.json")}
    common = sorted(set(wavs.keys()) & set(jsons.keys()))
    pairs = [(wavs[s], jsons[s]) for s in common]
    if len(pairs) < 2:
        raise ValueError(f"Expected at least 2 wav/json pairs in {conv_dir}")
    return pairs


def main(conv_dir: str) -> None:
    conv_path = Path(conv_dir).expanduser().resolve()
    if not conv_path.is_dir():
        raise ValueError(f"Conversation directory not found: {conv_path}")

    conv_id = conv_path.name
    out_root = conv_path / "transformed"
    out_audio = out_root / "audio_16k"
    out_vad = out_root / "vad"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_vad.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(conv_path)

    # Keep ordering consistent: sort by stem
    pairs = sorted(pairs, key=lambda x: x[0].stem)

    # Process each speaker
    vad_list = []
    resampled = []
    for wav_path, json_path in pairs[:2]:
        vad_list.append(read_vad_list(json_path))
        audio, sr = resample_wav(wav_path, TARGET_SR)
        resampled.append(audio)

    # Save combined VAD list (two channels)
    vad_path = out_vad / f"{conv_id}.json"
    with vad_path.open("w", encoding="utf-8") as f:
        json.dump(vad_list, f)

    # Save stereo wav (two channels)
    n = min(len(resampled[0]), len(resampled[1]))
    stereo = np.stack([resampled[0][:n], resampled[1][:n]], axis=1)
    stereo_path = out_audio / f"{conv_id}.wav"
    sf.write(str(stereo_path), stereo, TARGET_SR)

    print(f"Saved stereo 16k wav to: {stereo_path}")
    print(f"Saved VAD json to: {vad_path}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--conv_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.conv_dir)
