"""
Create train/val split files from the original MSR-VTT train metadata.

Workflow:
1) Place the official train JSON under data/msrvtt/raw (default: msrvtt_train_7k.json).
2) Run this script to produce train/val JSONs under data/msrvtt/splits.
3) Point configs to the generated files (train: msrvtt_train.json, val: msrvtt_val.json).
"""
import argparse
import json
import os
import random
from typing import List, Tuple


def load_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format in {path}")
    return data


def save_json(data: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def split_train_val(data: List[dict], val_ratio: float, seed: int) -> Tuple[List[dict], List[dict]]:
    if not 0 < val_ratio < 1:
        raise ValueError(f"val_ratio must be in (0,1); got {val_ratio}")
    if len(data) < 2:
        raise ValueError("Dataset too small to split")
    val_size = max(1, int(len(data) * val_ratio))
    val_size = min(val_size, len(data) - 1)

    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)

    val_idx = set(idx[:val_size])
    val_split = [data[i] for i in idx[:val_size]]
    train_split = [data[i] for i in idx[val_size:]]
    return train_split, val_split


def main(args):
    raw_train_path = args.raw_train
    # Backward-compatible fallback if user keeps JSON under splits
    if not os.path.exists(raw_train_path):
        alt = os.path.join("data/msrvtt/splits", os.path.basename(raw_train_path))
        if os.path.exists(alt):
            raw_train_path = alt
            print(f"Raw train JSON not found at {args.raw_train}, using {alt}")
        else:
            raise FileNotFoundError(f"Raw train JSON not found at {args.raw_train}")

    print(f"Loading train metadata from {raw_train_path}")
    data = load_list(raw_train_path)
    train_split, val_split = split_train_val(data, args.val_ratio, args.seed)
    print(f"Split {len(data)} items into {len(train_split)} train / {len(val_split)} val")

    train_out = os.path.join(args.out_dir, args.train_name)
    val_out = os.path.join(args.out_dir, args.val_name)
    save_json(train_split, train_out)
    save_json(val_split, val_out)
    print(f"Saved train split to {train_out}")
    print(f"Saved val split to {val_out}")

    # Optionally copy test metadata into splits for consistency
    if args.raw_test:
        raw_test_path = args.raw_test
        if not os.path.exists(raw_test_path):
            alt_test = os.path.join("data/msrvtt/splits", os.path.basename(raw_test_path))
            if os.path.exists(alt_test):
                raw_test_path = alt_test
                print(f"Raw test JSON not found at {args.raw_test}, using {alt_test}")
            else:
                raw_test_path = None
        if raw_test_path:
            test_out = os.path.join(args.out_dir, args.test_name)
            if os.path.exists(test_out) and not args.overwrite_test:
                print(f"Test split already exists at {test_out}; skipping copy (use --overwrite_test to replace)")
            else:
                test_data = load_list(raw_test_path)
                save_json(test_data, test_out)
                print(f"Copied test split to {test_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepare train/val splits from raw MSR-VTT train JSON")
    ap.add_argument("--raw_train", default="data/msrvtt/raw/msrvtt_train_7k.json", help="Path to raw train JSON")
    ap.add_argument("--raw_test", default="data/msrvtt/raw/msrvtt_test_1k.json", help="Path to raw test JSON (optional copy)")
    ap.add_argument("--out_dir", default="data/msrvtt/splits", help="Output directory for split JSONs")
    ap.add_argument("--train_name", default="msrvtt_train.json", help="Output filename for train split")
    ap.add_argument("--val_name", default="msrvtt_val.json", help="Output filename for val split")
    ap.add_argument("--test_name", default="msrvtt_test_1k.json", help="Output filename for test split (if copied)")
    ap.add_argument("--val_ratio", type=float, default=0.05, help="Fraction of raw train data to use for validation")
    ap.add_argument("--seed", type=int, default=2271, help="Random seed for shuffling before split")
    ap.add_argument("--overwrite_test", action="store_true", help="Overwrite test file if it already exists in out_dir")
    args = ap.parse_args()
    main(args)
