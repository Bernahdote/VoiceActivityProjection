from argparse import ArgumentParser
from pathlib import Path
from sklearn.model_selection import train_test_split
from vap.data.datamodule import load_df
from vap.utils.utils import read_txt


def process_file(args, split):
    file_path = getattr(args, f"{split}_file")
    session_names = read_txt(file_path)
    split_df = df[df["session"].isin(session_names)]
    r = len(split_df) / N
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    split_df.to_csv(f"{args.output_dir}/{split}.csv", index=False)
    print(f"Saved {split.capitalize()} file {len(session_names)}/{N} = {100*r:.1f}%")
    print(f"-> {args.output_dir}/{split}.csv")
    if len(split_df) != len(session_names):
        print(f"Warning: {len(split_df)} != {len(session_names)}")
        print(f"Missed {len(session_names) - len(split_df)} files")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/audio_vad.csv")
    parser.add_argument("--output_dir", type=str, default="data/splits")
    parser.add_argument("--train_size", type=float, default=0.8)
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--val_file", type=str, default=None)
    parser.add_argument("--test_file", type=str, default=None)

    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    df = load_df(args.csv)
    N = len(df)

    # Add session column
    df["session"] = df["audio_path_a"].apply(lambda x: Path(x).stem.rsplit("_", 1)[0]) # WB 
    # Extract domain from seamless_interaction path layout:
    # .../<domain>/<split>/<id1>/<id2>/<file>.wav
    df["domain"] = df["audio_path_a"].apply(lambda x: Path(x).parts[-5])


    # If any file path were provided we simply extract those
    if args.train_file or args.val_file or args.test_file:
        if args.train_file:
            process_file(args, "train")

        if args.val_file:
            process_file(args, "val")

        if args.test_file:
            process_file(args, "test")
    else:
        if args.train_size <= 0 or args.val_size <= 0:
            raise ValueError("train_size and val_size must be > 0.")
        if args.train_size + args.val_size >= 1.0:
            raise ValueError("train_size + val_size must be < 1.0.")

        # 1) Stratified split into train and remainder
        train_df, rest_df = train_test_split(
            df,
            train_size=args.train_size,
            random_state=0,
            stratify=df["domain"],
        )

        # 2) Split remainder into val/test to match requested absolute val_size
        rest_frac = 1.0 - args.train_size
        val_frac_within_rest = args.val_size / rest_frac
        val_df, test_df = train_test_split(
            rest_df,
            train_size=val_frac_within_rest,
            random_state=0,
            stratify=rest_df["domain"],
        )
        # Save splits
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        train_df.to_csv(f"{args.output_dir}/train.csv", index=False)
        val_df.to_csv(f"{args.output_dir}/val.csv", index=False)
        test_df.to_csv(f"{args.output_dir}/test.csv", index=False)
        print(f"Saved {len(train_df)} -> ", f"{args.output_dir}/train.csv")
        print(f"Saved {len(val_df)} -> ", f"{args.output_dir}/val.csv")
        print(f"Saved {len(test_df)} -> ", f"{args.output_dir}/test.csv")
