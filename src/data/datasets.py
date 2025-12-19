import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _uniform_sample(arr: np.ndarray, target_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniformly sample or pad an array to target_len along the first dimension.

    Returns sampled/padded array and a boolean mask (True = padding).
    """
    if arr.ndim == 1:
        arr = arr[:, None]
    T = arr.shape[0]
    if T >= target_len:
        idxs = np.linspace(0, T - 1, target_len).astype(int)
        sampled = arr[idxs]
        mask = np.zeros(target_len, dtype=bool)
    else:
        pad_len = target_len - T
        pad = np.zeros((pad_len, *arr.shape[1:]), dtype=arr.dtype)
        sampled = np.concatenate([arr, pad], axis=0)
        mask = np.zeros(target_len, dtype=bool)
        mask[T:] = True
    return sampled, mask


class MSRVTTFeatures(Dataset):
    """
    MSR-VTT dataset loader for pre-extracted 2D / 3D / object features.

    Expected layout for a split:
        features_root/
            2d/<video_id>.npy      # [T2d, D2]
            3d/<video_id>.npy      # [T3d, D3]
            obj/<video_id>.npy     # [Tobj, Dobj] (optional)
    """

    def __init__(
        self,
        split_json: str,
        features_root: str,
        tokenizer,
        max_frames: int = 16,
        use_3d: bool = True,
        use_obj: bool = True,
        caption_sampling: str = "first",
    ):
        if split_json is None or features_root is None:
            raise ValueError("split_json and features_root must both be provided")

        self.tok = tokenizer
        self.max_frames = max_frames
        self.use_3d = use_3d
        self.use_obj = use_obj
        self.caption_sampling = caption_sampling

        self.items = self._load_split_file(split_json)
        self.root = Path(features_root)
        self._attach_feature_paths()

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _load_split_file(split_json: str) -> List[dict]:
        with open(split_json, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = list(data.values())
        if not isinstance(data, list):
            raise ValueError(f"Unsupported split file format: {split_json}")

        cleaned = []
        for entry in data:
            video_id = (
                entry.get("video_id")
                or entry.get("videoid")
                or (f"video{entry['id']}" if "id" in entry else None)
            )
            if video_id is None:
                continue
            cleaned.append(
                {
                    "video_id": str(video_id),
                    "video": entry.get("video", ""),
                    "caption": entry.get("caption", ""),
                }
            )
        return cleaned

    def _attach_feature_paths(self):
        miss_2d, miss_3d = [], []
        for entry in self.items:
            vid = entry["video_id"].replace(".mp4", "")
            p2d = self.root / "2d" / f"{vid}.npy"
            p3d = self.root / "3d" / f"{vid}.npy"
            pobj = self.root / "obj" / f"{vid}.npy"

            if not p2d.exists():
                miss_2d.append(vid)
            entry["path_2d"] = str(p2d)

            if self.use_3d:
                if not p3d.exists():
                    miss_3d.append(vid)
                entry["path_3d"] = str(p3d)

            if self.use_obj and pobj.exists():
                entry["path_obj"] = str(pobj)

        errors = []
        if miss_2d:
            preview = ", ".join(miss_2d[:5])
            errors.append(f"{len(miss_2d)} missing 2D features (examples: {preview})")
        if self.use_3d and miss_3d:
            preview = ", ".join(miss_3d[:5])
            errors.append(f"{len(miss_3d)} missing 3D features (examples: {preview})")
        if errors:
            raise FileNotFoundError("; ".join(errors))

    def _select_caption(self, caption_field):
        if isinstance(caption_field, list):
            if self.caption_sampling == "random" and caption_field:
                return random.choice(caption_field)
            return caption_field[0] if caption_field else ""
        return caption_field

    def __getitem__(self, idx):
        it = self.items[idx]

        # 选出一条 caption
        caption = self._select_caption(it["caption"])

        # 统一成字符串，作为 raw_caption + tokenizer 输入
        if isinstance(caption, list):
            # 例如 ["a", "man", "is", "speaking"] 这种，就拼成一句话
            caption_text = " ".join(str(w) for w in caption)
        else:
            caption_text = str(caption)

        raw_caption = caption_text  # 保存原始文本形式，给 CLIP / 对比学习用

        # 加载特征
        feats2d = np.load(it["path_2d"])  # [T, D2]
        feats3d = np.load(it["path_3d"]) if self.use_3d else None
        obj_feats = np.load(it["path_obj"]) if self.use_obj and "path_obj" in it else None

        if obj_feats is not None:
            # If per-frame object list [T, K, D] average over K
            if obj_feats.ndim == 3:
                obj_feats = obj_feats.mean(axis=1)
            elif obj_feats.ndim != 2:
                raise ValueError(f"Unsupported object feature shape: {obj_feats.shape}")

        feats2d, mask2d = _uniform_sample(feats2d, self.max_frames)
        if feats3d is not None:
            feats3d, mask3d = _uniform_sample(feats3d, self.max_frames)
        else:
            feats3d = np.zeros((self.max_frames, 1), dtype=np.float32)
            mask3d = np.ones(self.max_frames, dtype=bool)

        if obj_feats is not None:
            obj_feats, mask_obj = _uniform_sample(obj_feats, self.max_frames)
        else:
            obj_feats, mask_obj = None, None

        # Final feat_mask: padding where any modality is padded
        feat_mask = mask2d | mask3d
        if mask_obj is not None:
            feat_mask = feat_mask | mask_obj
        feat_mask = torch.from_numpy(feat_mask)

        feats2d = torch.from_numpy(feats2d).float()
        feats3d = torch.from_numpy(feats3d).float()
        obj_tensor = torch.from_numpy(obj_feats).float() if obj_feats is not None else None

        # 用 caption_text 做 BPE 编码（而不是原始 list）
        tok = self.tok.encode(caption_text)
        tgt = torch.tensor(tok.ids, dtype=torch.long)

        return {
            "video_id": it["video_id"],
            "feats2d": feats2d,
            "feats3d": feats3d,
            "obj_feats": obj_tensor,
            "feat_mask": feat_mask,
            "tgt": tgt,
            "raw_caption": raw_caption,
        }
