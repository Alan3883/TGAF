#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize video frames aligned with feature importance curves.
Style: Contiguous frames (Panorama) + Temporal Curve.
Target: Academic Paper Quality (CVPR/ICCV style) - Balanced & Centered.
"""

import argparse
import os
import random
import math

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml
from PIL import Image

from src.data.datasets import MSRVTTFeatures
from src.utils.tokenizer import BPETokenizer
from src.utils.video import sample_video_frames

# -------------------------
# 1. Paper-Quality Style
# -------------------------
def setup_plot_style():
    """Config matplotlib for clean, serif-font academic figures."""
    try:
        import seaborn as sns
        sns.set_theme(style="ticks", context="paper", font="serif")
    except ImportError:
        pass

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": False, 
        "figure.constrained_layout.use": False,
    })

# -------------------------
# 2. Data Loading Helpers
# -------------------------
def split_root(cfg, name: str) -> str:
    roots = cfg.get("features_root")
    if roots and isinstance(roots, dict):
        return roots.get(name, cfg.get("features_dir"))
    return os.path.join(cfg["features_dir"], name)

def load_dataset_and_sample(cfg, split: str, index: int):
    tok_cfg = cfg["tokenizer"]
    tok = BPETokenizer(
        tok_cfg["model_path"],
        tok_cfg["bos_token"],
        tok_cfg["eos_token"],
        tok_cfg["pad_token"],
    )

    split_json = cfg["splits"][split]
    features_root = split_root(cfg, split)

    ds = MSRVTTFeatures(
        split_json=split_json,
        features_root=features_root,
        tokenizer=tok,
        max_frames=cfg["max_frames"],
        use_3d=cfg.get("use_3d", True),
        use_obj=cfg.get("use_obj", True),
    )

    if index < 0:
        index = random.randint(0, len(ds) - 1)
    
    sample = ds[index]
    meta = ds.items[index]
    return ds, sample, meta, index

def compute_importance(feats2d_np, feats3d_np, obj_np, mask_np):
    """Compute normalized L2 norm importance curve (valid steps only)."""
    if mask_np is None:
        return None, None, {}

    valid_idx = np.where(~mask_np)[0]
    if len(valid_idx) == 0:
        return None, None, {}

    comps = {}
    
    def get_norm(feats):
        if feats is None: return None
        f = feats[valid_idx]
        norms = np.linalg.norm(f, axis=1)
        if norms.max() > 1e-6:
            norms = norms / norms.max()
        else:
            norms = np.zeros_like(norms)
        return norms

    if feats2d_np is not None: comps["2d"] = get_norm(feats2d_np)
    if feats3d_np is not None: comps["3d"] = get_norm(feats3d_np)
    if obj_np is not None:     comps["obj"] = get_norm(obj_np)

    # Filter flat
    active = {k: v for k, v in comps.items() if (v.max() - v.min() > 1e-3)}
    
    if not active:
        # Fallback uniform
        return np.ones(len(valid_idx)) * 0.5, valid_idx, {}

    stack = np.stack(list(active.values()), axis=0)
    combined = stack.mean(axis=0)
    
    return combined, valid_idx, active

# -------------------------
# 3. Visualization Core
# -------------------------
def plot_panorama_compact(
    frames, 
    importance_curve, 
    per_modality,
    valid_indices,
    raw_caption,
    video_id,
    out_path,
    num_show=8
):
    """
    Plots frames stitched horizontally (Panorama) with an aligned curve below.
    Updates: Title padding, Center alignment, Bottom-Right Legend.
    """
    # --- 1. Downsample Frames ---
    total_valid = len(valid_indices)
    if total_valid <= num_show:
        show_idx_local = np.arange(total_valid)
    else:
        show_idx_local = np.linspace(0, total_valid - 1, num_show, dtype=int)
    
    frames_to_plot = [frames[i] for i in show_idx_local if i < len(frames)]
    time_points = [valid_indices[i] for i in show_idx_local if i < len(frames)]

    # --- 2. Figure Setup ---
    fig = plt.figure(figsize=(14, 4.0)) 
    
    # GridSpec: Ultra tight hspace to connect images with curve
    gs = gridspec.GridSpec(
        2,
        len(frames_to_plot),
        height_ratios=[1.3, 1.0],
        hspace=0.04,
        wspace=0.02
    )

    # --- 3. Plot Images ---
    for i, img in enumerate(frames_to_plot):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # Clean spines
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Subtle divider between frames
        if i > 0:
            ax.spines['left'].set_visible(True)
            ax.spines['left'].set_color('white')
            ax.spines['left'].set_linewidth(1.5)

    # --- 4. Plot Curve ---
    ax_curve = fig.add_subplot(gs[1, :])
    
    x_axis = valid_indices[:len(importance_curve)]
    
    # Standard Colors
    colors = {'2d': '#0072B2', '3d': '#E69F00', 'obj': '#009E73'} 
    
    # Plot modalities
    for mod_name, mod_curve in per_modality.items():
        if len(mod_curve) > len(x_axis):
            mod_curve = mod_curve[:len(x_axis)]
        ax_curve.plot(
            x_axis,
            mod_curve,
            label=mod_name.upper(), 
            linestyle='--',
            alpha=0.8,
            linewidth=1.5,
            color=colors.get(mod_name, 'gray')
        )

    # Plot Combined
    ax_curve.plot(
        x_axis,
        importance_curve,
        label="Combined",
        color='#D55E00',
        linewidth=2.5
    )
    ax_curve.fill_between(x_axis, importance_curve, color='#D55E00', alpha=0.15)

    # Vertical Markers (connecting frames to time steps)
    for t in time_points:
        ax_curve.axvline(x=t, color='black', linestyle=':', alpha=0.3, linewidth=1.0)
        if t in x_axis:
            idx_in_curve = np.where(x_axis == t)[0][0]
            val = importance_curve[idx_in_curve]
            ax_curve.plot(t, val, 'o', color='black', markersize=4, alpha=0.6)

    # Formatting Axis
    ax_curve.set_xlim(x_axis[0] - 0.5, x_axis[-1] + 0.5)
    ax_curve.set_ylim(-0.05, 1.15) 
    ax_curve.set_ylabel("Importance", fontweight='bold', labelpad=8)
    ax_curve.set_xlabel("Time Step (Frame Index)", fontweight='bold', labelpad=5)
    ax_curve.grid(True, linestyle=':', alpha=0.5)

    # --- 5. Legend: Bottom Right (Inside), Smaller & Tighter ---
    ax_curve.legend(
        loc='lower right',
        ncol=1,          # vertical stack
        frameon=True,
        facecolor='white',
        edgecolor='#e0e0e0',
        framealpha=0.95,
        fontsize=8,      # 更小的字体
        borderpad=0.3,   # 图例框内边距小一点
        handlelength=1.2,# 线段短一点
        handletextpad=0.4,
        labelspacing=0.2 # 各行之间距离小一点
    )

    # --- 6. Title: Centered with Breathing Room ---
    header_text = f"Video ID: {video_id}   |   Ground Truth: {raw_caption}"
    if len(header_text) > 130:
        header_text = header_text[:127] + "..."

    fig.text(
        0.5,
        0.93,                 # 你原来设的 0.93，这里保持
        header_text,
        fontsize=13,
        fontweight='bold', 
        ha='center',
        va='top',
        color='#222222'
    )

    # --- 7. Layout: Symmetrical & Spaced ---
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.15
    )

    # --- 8. Manually Narrow & Center the Curve Panel Horizontally ---
    # 在整体布局确定之后，对曲线这块单独做横向“缩窄 + 居中”
    box = ax_curve.get_position()  # [x0, y0, width, height] in figure coords
    ax_curve.set_position([0.09, box.y0, 0.86, box.height])
    # 0.15 + 0.70 + 0.15 = 1.0 → 居中

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300) 
    plt.close(fig)
    print(f"[Success] Saved visualization to: {out_path}")


# -------------------------
# 4. Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize features (Final Layout).")
    ap.add_argument("--config", type=str, default="configs/msrvtt_vit32.yaml")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--index", type=int, default=0, help="-1 for random")
    ap.add_argument("--video_root", type=str, default="data/msrvtt/raw/MSRVTT_Videos")
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--num_show", type=int, default=8, help="Number of frames to display")
    args = ap.parse_args()

    setup_plot_style()
    cfg = yaml.safe_load(open(args.config, "r"))

    # Load data
    try:
        ds, sample, meta, idx = load_dataset_and_sample(cfg, args.split, args.index)
    except Exception as e:
        print(f"[Error] Could not load dataset: {e}")
        return

    video_id = sample["video_id"]
    raw_caption = sample.get("raw_caption", "")
    video_file = meta.get("video", "") or f"{video_id}.mp4"
    
    print(f"Processing: {video_id} | Caption: {raw_caption}")

    # Prepare features
    feats2d = sample["feats2d"].cpu().numpy()
    feats3d = sample["feats3d"].cpu().numpy() if cfg.get("use_3d", False) else None
    obj_feats = sample["obj_feats"].cpu().numpy() if cfg.get("use_obj", False) and sample["obj_feats"] is not None else None
    mask = sample["feat_mask"].cpu().numpy().astype(bool)

    combined, valid_idx, per_modality = compute_importance(feats2d, feats3d, obj_feats, mask)

    if combined is None:
        print("[Skipping] No valid features.")
        return

    # Extract Video Frames
    video_path = os.path.join(args.video_root, video_file)
    if not os.path.exists(video_path):
        video_path = os.path.join(args.video_root, f"{video_id}.mp4")
    
    frames = []
    if os.path.exists(video_path):
        raw_frames = sample_video_frames(video_path, max_frames=cfg["max_frames"], fps=args.fps)
        valid_frames = []
        for i in valid_idx:
            if i < len(raw_frames):
                valid_frames.append(raw_frames[i])
        frames = valid_frames
    else:
        print(f"[Warning] Video not found: {video_path}")
        frames = [np.zeros((224, 224, 3), dtype=np.uint8) + 200 for _ in valid_idx]

    # Output
    base_out = args.out_dir or os.path.join(cfg.get("output_dir", "outputs"), "feature_viz_final_v2")
    out_name = f"{args.split}_{video_id}_viz.png"
    out_path = os.path.join(base_out, out_name)

    plot_panorama_compact(
        frames=frames,
        importance_curve=combined,
        per_modality=per_modality,
        valid_indices=valid_idx,
        raw_caption=raw_caption,
        video_id=video_id,
        out_path=out_path,
        num_show=args.num_show
    )

if __name__ == "__main__":
    main()
