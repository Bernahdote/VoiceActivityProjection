from os.path import join, isfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--audio_dir", type=str)
    parser.add_argument("--vad_dir", type=str)
    parser.add_argument("--output", type=str, default="data/audio_vad.csv")
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")

    audio_paths = list(Path(args.audio_dir).rglob("*.wav")) #

    json_paths = list(Path(args.vad_dir).rglob("*.json"))
    json_map = {p.stem: p for p in json_paths}


    groups = {} # WB
    data = []

    for audio_path in tqdm(audio_paths): # WB
        stem = audio_path.stem # WB
        conv_id = stem.rsplit("_", 1)[0] # WB. Stripping to conversation ID
        if conv_id not in groups: 
            groups[conv_id] = []
        groups[conv_id].append(audio_path) # WB. Searching mechanism -- might be a better solution 
    
    for conv_id, paths in groups.items(): # WB
        if len(paths) != 2: # WB
            raise ValueError(f"Expected exactly 2 ID's per conversation {conv_id}") # WB
        audio_a, audio_b = paths[:2] # WB
        vad_a = audio_a.with_suffix(".json") # WB -- Think this is correct
        vad_b = audio_b.with_suffix(".json") # WB

        data.append({ # WB
            "audio_path_a": str(audio_a), # WB
            "audio_path_b": str(audio_b), # WB
            "vad_path_a": str(vad_a), # WB
            "vad_path_b": str(vad_b), # WB

        })



        


    # data = []
    # skipped = []
    # for audio_path in tqdm(audio_paths):
    #     name = audio_path.stem

    #     conv_id = stem.rsplit("_", 1)[0] 
    #     vad_path = join(args.vad_dir, f"{name}.json")
    #     if not isfile(vad_path):
    #         # print(f"Missing {vad_path}")
    #         skipped.append(vad_path)
    #         continue
    #     data.append(
    #         {
    #             "audio_path": str(audio_path),
    #             "vad_path": vad_path,
    #         }
    #     )

    # if len(skipped) > 0:
    #     print("Skipped: ", len(skipped))
    #     with open("/tmp/create_audio_vad_json_errors.txt", "w") as f:
    #         f.write("\n".join(skipped))
    #     print("See -> /tmp/create_audio_vad_json_errors.txt")
    #     print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(args.output, index=False)
    print("Saved -> ", args.output)
