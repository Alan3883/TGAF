import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from src.utils.video import sample_video_frames
from src.utils.feature_extractor import CLIPViTFeatureExtractor


def load_split(json_file: str) -> List[dict]:
    """Load split annotations from JSON."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format: {json_file}")
    return data


def resolve_video_path(input_dir: Path, item: dict) -> Optional[Path]:
    """Resolve actual video path from split item."""
    vid = str(item.get("video_id", "")).replace(".mp4", "")
    video_name = item.get("video", f"{vid}.mp4")
    candidates = [
        input_dir / video_name,
        input_dir / f"{vid}.mp4",
        input_dir / vid / video_name,
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def crop_boxes(frame: np.ndarray, boxes: np.ndarray, max_objects: int) -> List[Image.Image]:
    """
    Crop up to max_objects boxes, filtering out tiny boxes.
    Falls back to full-frame if nothing valid remains.
    """
    MIN_SIZE = 16  # pixels
    h, w, _ = frame.shape
    crops = []
    # boxes: [N, 4] in xyxy
    for b in boxes[:max_objects]:
        x1, y1, x2, y2 = [int(v) for v in b]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        if bw < MIN_SIZE or bh < MIN_SIZE:
            continue
        crops.append(Image.fromarray(frame[y1:y2, x1:x2]).convert("RGB"))
    if not crops:
        crops = [Image.fromarray(frame).convert("RGB")]
    return crops


@torch.no_grad()
def extract_single_object_features(
    video_path: str,
    detector,
    clip_extractor,
    device,
    max_frames: int,
    fps: int,
    max_objects: int,
) -> Optional[np.ndarray]:
    """Run YOLO + CLIP on one video and return [T, D_obj] features."""
    frames = sample_video_frames(video_path, max_frames=max_frames, fps=fps)
    if not frames:
        return None

    feats_per_frame = []
    for frame in frames:
        # Normalize to HWC uint8 for detector/cropping
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        if frame.ndim == 3 and frame.shape[0] in (1, 3):  # CHW -> HWC
            frame = np.transpose(frame, (1, 2, 0))
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        if frame.ndim != 3 or frame.shape[2] not in (1, 3, 4):
            continue  # skip malformed frames
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # YOLO detection
        results = detector(frame, verbose=False)
        if not results:
            crops = []
        else:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            if len(boxes) > 0:
                order = np.argsort(-confs)
                boxes = boxes[order]
            crops = crop_boxes(frame, boxes, max_objects)

        if not crops:
            crops = [Image.fromarray(frame).convert("RGB")]

        # CLIP encoding of cropped objects
        try:
            obj_feats = clip_extractor(crops)  # [K, D]
        except Exception as e:  # pylint: disable=broad-except
            print(f"Warning: CLIP failed on {video_path}: {e}. Falling back to full frame.")
            obj_feats = clip_extractor([Image.fromarray(frame).convert("RGB")])

        obj_vec = obj_feats.mean(dim=0)  # [D]
        feats_per_frame.append(obj_vec.cpu().numpy())

    return np.stack(feats_per_frame, axis=0)  # [T, D_obj]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # YOLO detector
    detector = YOLO(args.yolo_model)

    # CLIP visual encoder with fast image processor
    clip_extractor = CLIPViTFeatureExtractor(
        model_name=args.clip_model,
        use_fast=True,  # <-- new flag, see CLIPViTFeatureExtractor implementation
    ).to(device).eval()

    items = load_split(args.json_file)
    out_dir = Path(args.out_dir) / "obj"
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for item in tqdm(items, desc="Extracting object features"):
        vid = str(item.get("video_id", "")).replace(".mp4", "")
        video_path = resolve_video_path(Path(args.input_dir), item)
        if video_path is None:
            print(f"Warning: video not found for {vid}")
            skipped += 1
            continue

        out_path = out_dir / f"{vid}.npy"
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        feats = extract_single_object_features(
            str(video_path),
            detector,
            clip_extractor,
            device,
            max_frames=args.max_frames,
            fps=args.fps,
            max_objects=args.max_objects,
        )
        if feats is None:
            skipped += 1
            continue

        np.save(out_path, feats)
        processed += 1

    print(f"\nDone. processed={processed}, skipped={skipped}, out_dir={out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract object-level features with YOLO + CLIP")
    ap.add_argument("--input_dir", required=True, help="Directory with raw videos")
    ap.add_argument("--json_file", required=True, help="Split JSON file")
    ap.add_argument("--out_dir", required=True, help="Output split directory, e.g., data/msrvtt/features/train")
    ap.add_argument("--max_frames", type=int, default=16, help="Max frames per video")
    ap.add_argument("--fps", type=int, default=2, help="Sampling FPS")
    ap.add_argument("--yolo_model", default="yolov8x.pt", help="YOLO model name/path")
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32", help="CLIP model id")
    ap.add_argument("--max_objects", type=int, default=5, help="Max objects per frame")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()
    main(args)
