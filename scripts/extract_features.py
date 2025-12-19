import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils.video import sample_video_frames
from src.utils.feature_extractor import create_feature_extractor


@torch.no_grad()
def extract_single(video_path: str, extractor, device, max_frames: int, fps: int):
    """
    Sample frames from a video and extract features.

    Args:
        video_path: Path to the raw video file.
        extractor: Feature extractor module.
        device: torch.device.
        max_frames: Max number of frames to sample.
        fps: Sampling FPS.

    Returns:
        Numpy array of shape [T, D] or None if no frames.
    """
    frames = sample_video_frames(video_path, max_frames=max_frames, fps=fps)
    if not frames:
        print(f"Warning: no frames extracted from {video_path}")
        return None
    feats = extractor(frames.to(device) if hasattr(frames, "to") else frames)
    return feats.detach().cpu().numpy()


def load_items(json_file: str):
    """
    Load split JSON. Supports list or dict-of-items format.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format in {json_file}")
    return data


def resolve_video_id(item: dict) -> str:
    """
    Make video id consistent with MSRVTTFeatures dataset logic.

    Priority: video_id -> videoid -> 'video{id}'.
    Strips '.mp4' suffix if present.
    """
    raw_id = (
        item.get("video_id")
        or item.get("videoid")
        or (f"video{item['id']}" if "id" in item else None)
    )
    if raw_id is None:
        raise ValueError(f"Missing video id in item: {item}")
    vid = str(raw_id)
    # remove trailing .mp4 if included in id
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build kwargs for feature extractor
    extractor_kwargs = {}
    if args.backbone in {"clip-vit", "clip"}:
        extractor_kwargs["model_name"] = args.clip_model
    if args.no_pretrained:
        extractor_kwargs["pretrained"] = False  # default is True in our extractors

    extractor = create_feature_extractor(args.backbone, **extractor_kwargs)
    extractor = extractor.to(device).eval()
    print(f"Loaded feature extractor: {args.backbone}")

    # Decide modality subfolder: 2d or 3d
    modality = "3d" if args.backbone in {"swin3d", "video-swin"} else "2d"
    out_root = Path(args.out_dir) / modality
    os.makedirs(out_root, exist_ok=True)
    print(f"Saving features to: {out_root}")

    # Determine video list JSON
    if args.json_file is None:
        candidates = [
            "data/msrvtt/splits/msrvtt_train.json",
            "data/msrvtt/splits/msrvtt_train_7k.json",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                args.json_file = cand
                print(f"Auto-detected JSON file: {args.json_file}")
                break
    if args.json_file is None:
        raise ValueError("Please provide --json_file or place a train split under data/msrvtt/splits/")

    items = load_items(args.json_file)
    print(f"Loaded {len(items)} items from {args.json_file}")

    processed = 0
    skipped = 0

    for item in tqdm(items, desc="Extracting features"):
        # robust video id
        vid = resolve_video_id(item)

        # resolve actual video file path
        video_name = item.get("video", f"{vid}.mp4")
        candidates = [
            Path(args.input_dir) / video_name,
            Path(args.input_dir) / f"{vid}.mp4",
        ]
        video_path = None
        for c in candidates:
            if c.exists():
                video_path = c
                break

        if video_path is None:
            print(f"Warning: video not found for id={vid} under {args.input_dir}")
            skipped += 1
            continue

        out_path = out_root / f"{vid}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        feats = extract_single(str(video_path), extractor, device, args.max_frames, args.fps)
        if feats is None:
            skipped += 1
            continue

        np.save(out_path, feats)
        processed += 1

    print(f"\nExtraction complete! processed={processed}, skipped={skipped}, out_root={out_root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract visual features from videos")
    ap.add_argument("--input_dir", default="data/msrvtt/raw/MSRVTT_Videos",
                    help="Directory with video files")
    ap.add_argument("--json_file", default=None,
                    help="Split JSON with video list (auto-detects train split if omitted)")
    ap.add_argument("--out_dir", default="data/msrvtt/features/train",
                    help="Output directory root for features (train/val/test)")
    ap.add_argument("--max_frames", type=int, default=16,
                    help="Max frames per video")
    ap.add_argument("--fps", type=int, default=2,
                    help="Sampling FPS")
    ap.add_argument("--backbone", default="resnet50",
                    choices=["resnet50", "resnet101", "clip-vit", "clip", "swin3d"],
                    help="Backbone for feature extraction")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32",
                    help="HF model id for CLIP (when backbone=clip-vit/clip)")
    ap.add_argument("--no_pretrained", action="store_true",
                    help="Do not load pretrained weights")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing feature files")
    args = ap.parse_args()
    main(args)
