import argparse
import os

import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from src.data.datasets import MSRVTTFeatures
from src.data.collate import collate_fn
from src.models.model import VideoCaptionModel
from src.utils.tokenizer import BPETokenizer


# ---------------------------------------------------------
# Helper: load config, model, tokenizer, and one test sample
# ---------------------------------------------------------
def load_model_and_sample(config_path, checkpoint_path, video_idx: int):
    """
    Load model + tokenizer + the test sample at index video_idx.
    video_idx is the index in the test split: [0, len(test_ds)-1].
    """
    cfg = yaml.safe_load(open(config_path, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # tokenizer
    tok = BPETokenizer(
        cfg["tokenizer"]["model_path"],
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"],
        cfg["tokenizer"]["pad_token"],
    )
    pad_id = tok.pad_id

    # feature root helper
    def split_root(name: str) -> str:
        roots = cfg.get("features_root")
        if roots and isinstance(roots, dict):
            return roots.get(name, cfg.get("features_dir"))
        return os.path.join(cfg["features_dir"], name)

    # test dataset
    test_ds = MSRVTTFeatures(
        split_json=cfg["splits"]["test"],
        features_root=split_root("test"),
        tokenizer=tok,
        max_frames=cfg["max_frames"],
        use_3d=cfg.get("use_3d", True),
        use_obj=cfg.get("use_obj", True),
    )

    if not (0 <= video_idx < len(test_ds)):
        raise IndexError(f"video_idx {video_idx} out of range [0, {len(test_ds)-1}]")

    # small dataloader just to reuse collate_fn
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # get the batch at video_idx
    batch = None
    for i, b in enumerate(test_dl):
        if i == video_idx:
            batch = b
            break
    assert batch is not None, "Failed to load sample batch."

    (
        feats2d,
        feats3d,
        obj_feats,
        feat_mask,
        y_in,
        y_out,
        tgt_mask,
        raw_caps,
    ) = batch

    feats2d = feats2d.to(device)
    feats3d = feats3d.to(device)
    feat_mask = feat_mask.to(device)
    obj_feats = obj_feats.to(device) if obj_feats is not None else None

    # video id string (e.g., "video7020")
    video_id = test_ds.items[video_idx]["video_id"]

    # -------- build model + load checkpoint --------
    checkpoint_path = checkpoint_path or os.path.join(cfg["output_dir"], "last.pt")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # align_dim must match training
    align_dim = cfg["model"]["d_model"]
    if "align_proj.weight" in state:
        align_dim = state["align_proj.weight"].shape[1]
    print(f"[visual_gates] Using align_dim={align_dim}")

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
    print(f"[visual_gates] Loaded checkpoint from {checkpoint_path}")
    if missing:
        print(f"[visual_gates] Missing keys (OK if align-related): {missing}")
    if unexpected:
        print(f"[visual_gates] Unexpected keys: {unexpected}")

    model.eval()

    return (
        model,
        tok,
        feats2d,
        feats3d,
        obj_feats,
        feat_mask,
        video_id,
    )


# ---------------------------------------------------------
# Greedy decode using pre-computed 2D/3D memories
# ---------------------------------------------------------
@torch.no_grad()
def greedy_decode_with_mem(model, tok, mem_2d, mem_3d, feat_mask, max_len: int = 40):
    """
    Greedy decoding using pre-computed encoder outputs mem_2d / mem_3d.
    Returns:
      seq_ids: list of token ids including BOS/EOS.
    """
    device = mem_2d.device
    bos_id, eos_id = tok.bos_id, tok.eos_id

    seq = torch.full((1, 1), bos_id, device=device, dtype=torch.long)  # [1,1]

    for _ in range(max_len - 1):
        logits = model.decoder(
            seq,
            mem_2d=mem_2d,
            mem_3d=mem_3d,
            tgt_key_padding_mask=None,
            mem_2d_key_padding_mask=feat_mask,
            mem_3d_key_padding_mask=feat_mask,
        )  # [1, L, V]

        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
        seq = torch.cat([seq, next_id], dim=1)                   # [1, L+1]

        if next_id.item() == eos_id:
            break

    return seq[0].cpu().tolist()  # [L]


# ---------------------------------------------------------
# Collect gate maps for a given sequence
# ---------------------------------------------------------
@torch.no_grad()
def get_gate_for_sequence(
    model,
    tok,
    mem_2d,
    mem_3d,
    feat_mask,
    seq_ids,
    layer_idx: int = -1,
):
    """
    Run decoder once with teacher forcing on seq_ids and
    collect 2D/3D gate weights for each token from a given layer.

    Returns:
      words: [L_tokens] list of strings
      g2d:   [L_tokens] numpy array, gate weight for 2D
      g3d:   [L_tokens] numpy array, gate weight for 3D
    """
    device = mem_2d.device
    seq = torch.tensor(seq_ids, device=device, dtype=torch.long).unsqueeze(0)  # [1, L]

    # forward with gate maps from decoder
    logits, gate_maps = model.decoder(
        seq,
        mem_2d=mem_2d,
        mem_3d=mem_3d,
        tgt_key_padding_mask=None,
        mem_2d_key_padding_mask=feat_mask,
        mem_3d_key_padding_mask=feat_mask,
        return_gate_maps=True,   # << 只在可视化脚本里打开
    )  # logits: [1,L,V], gate_maps: list[num_layers] of [1,L]

    num_layers = len(gate_maps)
    if layer_idx < 0:
        layer = num_layers + layer_idx
    else:
        layer = layer_idx
    layer = max(0, min(layer, num_layers - 1))

    # gate for chosen layer: [L], already averaged over channels in decoder
    gate = gate_maps[layer][0]              # [L]
    gate = gate.detach().cpu().numpy()      # 2D weight

    bos_id, eos_id, pad_id = tok.bos_id, tok.eos_id, tok.pad_id
    ids = seq_ids
    g = gate

    # drop leading BOS
    if ids and ids[0] == bos_id:
        ids = ids[1:]
        g = g[1:]

    # cut at EOS
    if eos_id in ids:
        eos_pos = ids.index(eos_id)
        ids = ids[:eos_pos]
        g = g[:eos_pos]

    # drop PAD
    clean_ids = []
    clean_g = []
    for tid, w in zip(ids, g):
        if tid == pad_id:
            continue
        clean_ids.append(tid)
        clean_g.append(w)

    clean_g = np.array(clean_g, dtype=np.float32)

    # decode per token for display
    words = [tok.decode([tid]) for tid in clean_ids]
    g2d = clean_g
    g3d = 1.0 - clean_g

    return words, g2d, g3d


# ---------------------------------------------------------
# Plot function
# ---------------------------------------------------------
def plot_gates(words, g2d, g3d, out_path, title="2D vs 3D gate per token"):
    """
    Make a stacked bar plot: x-axis = tokens, y-axis = gate weights.
    """
    if len(words) == 0:
        print("[visual_gates] Empty caption, nothing to plot.")
        return

    x = np.arange(len(words))

    # figure width adapts to caption length
    width = max(6.0, len(words) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4))

    ax.bar(x, g2d, label="2D (appearance)")
    ax.bar(x, g3d, bottom=g2d, label="3D (motion)")

    ax.set_xticks(x)
    ax.set_xticklabels(words, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Gate weight")
    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[visual_gates] Saved gate visualization to {out_path}")


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize 2D/3D gate per token")
    ap.add_argument("--config", required=True, help="Config YAML path")
    ap.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    ap.add_argument(
        "--video_idx",
        type=int,
        required=True,
        help="Index in test split (0-based)",
    )
    ap.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Decoder layer index (0..L-1, negative = from last)",
    )
    ap.add_argument(
        "--max_len",
        type=int,
        default=40,
        help="Max decoding length",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output PNG path for visualization",
    )
    args = ap.parse_args()

    (
        model,
        tok,
        feats2d,
        feats3d,
        obj_feats,
        feat_mask,
        video_id,
    ) = load_model_and_sample(args.config, args.checkpoint, args.video_idx)

    # encode video into 2D/3D streams (uses FiLM + TCN etc.)
    with torch.no_grad():
        h2d, h3d, _, _ = model._encode_streams(
            feats2d,
            feats3d,
            feat_mask,
            obj_feats,
        )  # [1, T, d_model] each

    # greedy decode once to get full caption
    seq_ids = greedy_decode_with_mem(
        model,
        tok,
        h2d,
        h3d,
        feat_mask,
        max_len=args.max_len,
    )

    # get per-token gates from given layer
    words, g2d, g3d = get_gate_for_sequence(
        model,
        tok,
        h2d,
        h3d,
        feat_mask,
        seq_ids,
        layer_idx=args.layer,
    )

    title = f"2D vs 3D gate per token\nvideo={video_id}, layer={args.layer}"
    plot_gates(words, g2d, g3d, args.out, title=title)


if __name__ == "__main__":
    main()
