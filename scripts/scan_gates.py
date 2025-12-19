# scripts/scan_gates.py

import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import MSRVTTFeatures
from src.data.collate import collate_fn
from src.models.model import VideoCaptionModel
from src.utils.tokenizer import BPETokenizer


def build_feature_root(cfg, split_name: str) -> str:
    """
    Resolve feature root path for a given split name ("train"/"val"/"test").
    """
    roots = cfg.get("features_root")
    if roots and isinstance(roots, dict):
        return roots.get(split_name, cfg.get("features_dir"))
    return os.path.join(cfg["features_dir"], split_name)


def build_model_and_data(config_path: str, checkpoint_path: str):
    """
    Load config, tokenizer, test dataset and model (with correct align_dim).
    """
    cfg = yaml.safe_load(open(config_path, "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scan_gates] Using device: {device}")

    # tokenizer
    tok = BPETokenizer(
        cfg["tokenizer"]["model_path"],
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"],
        cfg["tokenizer"]["pad_token"],
    )
    pad_id = tok.pad_id

    # test dataset
    test_ds = MSRVTTFeatures(
        split_json=cfg["splits"]["test"],
        features_root=build_feature_root(cfg, "test"),
        tokenizer=tok,
        max_frames=cfg["max_frames"],
        use_3d=cfg.get("use_3d", True),
        use_obj=cfg.get("use_obj", True),
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # checkpoint + align_dim (must match training)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    align_dim = cfg["model"]["d_model"]
    if "align_proj.weight" in state:
        align_dim = state["align_proj.weight"].shape[1]
        print(f"[scan_gates] Using align_dim={align_dim}")
    else:
        print("[scan_gates] No align_proj in checkpoint, using d_model as align_dim")

    # model
    model = VideoCaptionModel(
        vocab_size=tok.vocab_size(),
        feature_dim_2d=cfg["feature_dim_2d"],
        feature_dim_3d=cfg["feature_dim_3d"],
        feature_dim_obj=cfg.get("feature_dim_obj", 0),
        d_model=cfg["model"]["d_model"],
        tcn_layers=cfg["encoder"]["layers"],
        k=cfg["encoder"]["kernel_size"],
        dilations=cfg["encoder"]["dilations"],
        dec_layers=cfg["decoder"]["num_layers"],
        nhead=cfg["decoder"]["nhead"],
        ff=cfg["decoder"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        pad_id=pad_id,
        align_dim=align_dim,
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[scan_gates] Loaded checkpoint from {checkpoint_path}")
    if missing:
        print(f"[scan_gates] Missing keys (OK if align-related): {missing}")
    if unexpected:
        print(f"[scan_gates] Unexpected keys: {unexpected}")

    model.eval()
    return cfg, tok, model, test_ds, test_dl


@torch.no_grad()
def greedy_decode_with_gates(
    model,
    tok,
    feats2d,
    feats3d,
    obj_feats,
    feat_mask,
    layer_idx: int,
    max_len: int,
):
    """
    Greedy decode one sample and return:
      - decoded text
      - gate_2d (per token, after BOS/EOS trim) as 1D tensor
      - gate_3d (1 - gate_2d)
      - full token id sequence
    """
    device = feats2d.device
    bos_id, eos_id, pad_id = tok.bos_id, tok.eos_id, tok.pad_id

    # 1) encode video streams once (same as in visual_gates)
    h2d, h3d, obj_aligned, obj_temporal = model._encode_streams(
        feats2d, feats3d, feat_pad_mask=feat_mask, obj_feats=obj_feats
    )

    # 2) greedy decode to get a caption
    seq = torch.full((1, 1), bos_id, dtype=torch.long, device=device)  # [1,1]
    for _ in range(max_len - 1):
        # normal decoder forward, no gate maps
        logits = model.decoder(
            seq,
            mem_2d=h2d,
            mem_3d=h3d,
            tgt_key_padding_mask=None,
            mem_2d_key_padding_mask=feat_mask,
            mem_3d_key_padding_mask=feat_mask,
        )
        next_logits = logits[:, -1, :]  # [1,V]
        next_id = next_logits.argmax(dim=-1, keepdim=True)  # [1,1]
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    # 3) run one more forward with return_gate_maps=True to collect gates
    logits, gate_maps = model.decoder(
        seq,
        mem_2d=h2d,
        mem_3d=h3d,
        tgt_key_padding_mask=None,
        mem_2d_key_padding_mask=feat_mask,
        mem_3d_key_padding_mask=feat_mask,
        return_gate_maps=True,   # scalar gate per token per layer
    )

    token_ids = seq[0].tolist()  # length L

    if gate_maps is None or len(gate_maps) == 0:
        return "", torch.empty(0), torch.empty(0), token_ids

    # gate_maps: list[num_layers] of [B,L] (2D weight)
    num_layers = len(gate_maps)
    if layer_idx < 0:
        layer = num_layers + layer_idx
    else:
        layer = layer_idx
    layer = max(0, min(layer, num_layers - 1))

    g2d_full = gate_maps[layer][0]  # [L]  (already averaged over channels)
    g3d_full = 1.0 - g2d_full       # [L]

    # trim BOS / EOS
    L = len(token_ids)
    start = 1  # skip BOS
    end = L
    if eos_id in token_ids:
        eos_pos = token_ids.index(eos_id)
        end = eos_pos  # do not include EOS itself

    if end <= start:
        g2d_trim = torch.empty(0, device=device)
        g3d_trim = torch.empty(0, device=device)
    else:
        g2d_trim = g2d_full[start:end]
        g3d_trim = g3d_full[start:end]

    # decode text (drop BOS/EOS/PAD)
    clean_ids = [tid for tid in token_ids if tid not in {bos_id, eos_id, pad_id}]
    text = tok.decode(clean_ids)

    return text, g2d_trim, g3d_trim, token_ids


def main():
    ap = argparse.ArgumentParser(
        description="Scan test split and find sentences with strongest 2D/3D gate bias."
    )
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint")
    ap.add_argument(
        "--out",
        default="outputs/scan_gates_top.txt",
        help="Output text file with top examples",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Decoder layer index for gate visualization (same as visual_gates)",
    )
    ap.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="How many strongest examples to keep",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Max decode length (default: use cfg['decoder']['max_len'])",
    )
    args = ap.parse_args()

    cfg, tok, model, test_ds, test_dl = build_model_and_data(
        args.config, args.checkpoint
    )

    max_len = args.max_len or cfg["decoder"]["max_len"]
    print(
        f"[scan_gates] Using max_len={max_len}, layer={args.layer}, "
        f"top_k={args.top_k}"
    )

    device = next(model.parameters()).device

    examples = []  # list of dicts

    for idx, batch in enumerate(tqdm(test_dl, desc="Scanning gates")):
        (
            feats2d,
            feats3d,
            obj_feats,
            feat_mask,
            _y_in,
            _y_out,
            _tgt_mask,
            _raw_caps,
        ) = batch

        feats2d = feats2d.to(device)
        feats3d = feats3d.to(device)
        feat_mask = feat_mask.to(device)
        obj_feats = obj_feats.to(device) if obj_feats is not None else None

        text, g2d, g3d, token_ids = greedy_decode_with_gates(
            model,
            tok,
            feats2d,
            feats3d,
            obj_feats,
            feat_mask,
            layer_idx=args.layer,
            max_len=max_len,
        )

        if g2d.numel() == 0:
            continue

        # score = average |g2d - 0.5|, larger = stronger 2D/3D bias
        score = torch.mean(torch.abs(g2d - 0.5)).item()

        video_id = (
            test_ds.items[idx]["video_id"]
            if hasattr(test_ds, "items")
            else f"video_{idx}"
        )

        examples.append(
            {
                "idx": idx,
                "video_id": video_id,
                "score": score,
                "caption": text,
            }
        )

    # sort by score desc
    examples.sort(key=lambda x: x["score"], reverse=True)
    top = examples[: args.top_k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rank, ex in enumerate(top, start=1):
            line = (
                f"[{rank}] idx={ex['idx']}, video_id={ex['video_id']}, "
                f"score={ex['score']:.4f}\n"
                f"    caption: {ex['caption']}\n\n"
            )
            f.write(line)

    print(f"[scan_gates] Saved top-{args.top_k} gate examples to {args.out}")
    print("Top examples (for quick view):")
    for rank, ex in enumerate(top, start=1):
        print(
            f"[{rank}] idx={ex['idx']}, video_id={ex['video_id']}, "
            f"score={ex['score']:.4f}, caption={ex['caption']}"
        )


if __name__ == "__main__":
    main()
