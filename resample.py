## WB


import subprocess
from pathlib import Path 

ROOT = Path("/mnt/sda/willem/datasets/seamless_interaction") #Linux
#ROOT = Path("/Users/willemberner/datasets/seamless_interaction") #Mac

# Optional: Remove MP3's for reducing storage. 

sr = 16000 
PCM_CODEC = "pcm_s16le"

def resample_file(path: Path): 
    tmp = path.with_name(f"{path.stem}.tmp.wav")

    cmd = ["ffmpeg", "-y", "-i", str(path), "-ar", str(sr), "-c:a", PCM_CODEC, str(tmp)]

    subprocess.run(cmd, check=True)
    tmp.replace(path)  


def main(): 
    wavs = list(ROOT.rglob("*.wav"))
    for i, wav in enumerate(wavs): 
        resample_file(wav)


if __name__ == "__main__":
    main()
