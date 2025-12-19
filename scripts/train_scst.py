import argparse
import os
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
import math

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.datasets import MSRVTTFeatures
from src.data.collate import collate_fn as train_collate_fn
from src.models.model import VideoCaptionModel
from src.utils.tokenizer import BPETokenizer


# ---------------------------------------------------------
# XE loss (same as train.py)
# ---------------------------------------------------------
def xe_loss(logits, y_out, pad_id):
    """
    Caption cross-entropy loss with label smoothing.
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y_out.view(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )


# ---------------------------------------------------------
# CIDEr references + our own CIDEr implementation
# ---------------------------------------------------------
def _normalize_vid_key(key) -> str:
    """
    Normalize video id:
      - int  -> "videoXXXX"
      - str  -> strip ".mp4"
    """
    if isinstance(key, int):
        vid = f"video{key}"
    else:
        vid = str(key)
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def load_multi_refs(pkl_path: str) -> Dict[str, List[str]]:
    """
    Load msrvtt_raw_captions pkl and convert to:
        refs[video_id] = ["sentence 1", "sentence 2", ...]
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find raw captions at {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    refs: Dict[str, List[str]] = {}

    # dict format
    if isinstance(data, dict):
        for k, v in data.items():
            vid = _normalize_vid_key(k)
            caps: List[str] = []

            if isinstance(v, list):
                # list[list[token]] or list[str]
                for entry in v:
                    if isinstance(entry, list):
                        words = [w for w in entry if isinstance(w, str)]
                        if not words:
                            continue
                        s = " ".join(words).strip()
                        if s:
                            caps.append(s)
                    elif isinstance(entry, str):
                        s = entry.strip()
                        if s:
                            caps.append(s)

            elif isinstance(v, dict):
                raw = v.get("captions") or v.get("sentences") or v.get("caption")
                if raw is None:
                    continue
                if isinstance(raw, str):
                    raw = [raw]
                if isinstance(raw, list):
                    for entry in raw:
                        if isinstance(entry, list):
                            words = [w for w in entry if isinstance(w, str)]
                            if not words:
                                continue
                            s = " ".join(words).strip()
                            if s:
                                caps.append(s)
                        elif isinstance(entry, str):
                            s = entry.strip()
                            if s:
                                caps.append(s)

            if caps:
                refs[vid] = caps

    # list[dict] format
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            vid_key = item.get("video_id") or item.get("videoid") or item.get("id")
            if vid_key is None:
                continue
            vid = _normalize_vid_key(vid_key)
            raw = item.get("captions") or item.get("sentences") or item.get("caption")
            if raw is None:
                continue
            if isinstance(raw, str):
                raw = [raw]

            caps: List[str] = []
            if isinstance(raw, list):
                for entry in raw:
                    if isinstance(entry, list):
                        words = [w for w in entry if isinstance(w, str)]
                            # 注意这里换行对齐
                        if not words:
                            continue
                        s = " ".join(words).strip()
                        if s:
                            caps.append(s)
                    elif isinstance(entry, str):
                        s = entry.strip()
                        if s:
                            caps.append(s)
            if caps:
                refs[vid] = caps

    print(f"[SCST] Loaded CIDEr references for {len(refs)} videos from {pkl_path}")
    return refs


# ---------- our own CIDEr (single-image) implementation ----------

def _precook(s: str, n: int = 4):
    """
    Turn a sentence into n-gram counts.
    """
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def build_cider_df(refs: Dict[str, List[str]], n: int = 4):
    """
    Build global document frequency for CIDEr from all reference captions.
    """
    df = defaultdict(float)
    # treat each video as one "document": union of ngrams across its refs
    for caps in refs.values():
        ngrams_video = set()
        for s in caps:
            if not isinstance(s, str):
                continue
            s = s.strip()
            if not s:
                continue
            cnts = _precook(s, n=n)
            ngrams_video.update(cnts.keys())
        for ng in ngrams_video:
            df[ng] += 1.0

    # N = number of "documents" (videos)
    N = max(1, len(refs))
    ref_len = math.log(float(N))
    print(f"[SCST] Built CIDEr DF with {len(df)} n-grams, ref_len={ref_len:.4f}, N_docs={N}")
    return df, ref_len


def _counts2vec(cnts, df, ref_len, n: int = 4):
    """
    Map n-gram counts to tf-idf vector.
    """
    vec = [defaultdict(float) for _ in range(n)]
    norm = [0.0 for _ in range(n)]
    length = 0

    for ngram, term_freq in cnts.items():
        df_ng = df.get(ngram, 1.0)
        idf = ref_len - math.log(df_ng)
        n_idx = len(ngram) - 1
        w = float(term_freq) * idf
        vec[n_idx][ngram] = w
        norm[n_idx] += w * w
        if n_idx == 1:
            length += term_freq

    norm = [math.sqrt(v) if v > 0.0 else 1.0 for v in norm]
    return vec, norm, length


def _sim(vec_h, vec_r, norm_h, norm_r, length_h, length_r, n: int = 4):
    """
    Cosine similarity over n-gram tf-idf vectors.
    """
    val = [0.0 for _ in range(n)]
    for i in range(n):
        for ngram, w_h in vec_h[i].items():
            w_r = vec_r[i].get(ngram, 0.0)
            val[i] += w_h * w_r
        if norm_h[i] > 0 and norm_r[i] > 0:
            val[i] /= (norm_h[i] * norm_r[i])
    return val


def compute_cider_reward_single(
    hyp: str,
    refs_sent: List[str],
    df,
    ref_len,
    n: int = 4,
) -> float:
    """
    Compute CIDEr reward for one hypothesis vs multi references.
    """
    if not isinstance(hyp, str):
        hyp = str(hyp)
    hyp = hyp.strip()
    if not hyp:
        return 0.0

    hyp_cnts = _precook(hyp, n=n)
    vec_h, norm_h, len_h = _counts2vec(hyp_cnts, df, ref_len, n=n)

    scores = []
    valid_refs = 0
    for r in refs_sent:
        if not isinstance(r, str):
            r = str(r)
        r = r.strip()
        if not r:
            continue
        ref_cnts = _precook(r, n=n)
        vec_r, norm_r, len_r = _counts2vec(ref_cnts, df, ref_len, n=n)
        val = _sim(vec_h, vec_r, norm_h, norm_r, len_h, len_r, n=n)
        score_ref = sum(val) / float(n)
        scores.append(score_ref)
        valid_refs += 1

    if valid_refs == 0:
        return 0.0

    score = sum(scores) / float(valid_refs)
    return float(score * 10.0)


def decode_tokens(token_ids: List[int], tok: BPETokenizer) -> str:
    """
    Decode token ids to plain text (remove BOS/EOS/PAD).
    """
    special = {tok.bos_id, tok.eos_id, tok.pad_id}
    clean = [tid for tid in token_ids if tid not in special]
    return tok.decode(clean)


# ---------------------------------------------------------
# decoding functions
# ---------------------------------------------------------
@torch.no_grad()
def greedy_decode(model, feats2d, feats3d, obj_feats, feat_mask, tok, max_len: int):
    """
    Greedy decoding as baseline (no gradient).
    """
    device = feats2d.device
    bos_id, eos_id = tok.bos_id, tok.eos_id

    was_training = model.training
    model.eval()

    seq = torch.full((1, 1), bos_id, device=device, dtype=torch.long)
    for _ in range(max_len - 1):
        logits = model(
            feats2d,
            feats3d,
            seq,
            feat_pad_mask=feat_mask,
            obj_feats=obj_feats,
        )
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1,1]
        seq = torch.cat([seq, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    if was_training:
        model.train()

    return seq[0].cpu().tolist()


def sample_decode(
    model,
    feats2d,
    feats3d,
    obj_feats,
    feat_mask,
    tok,
    max_len: int,
    topk: int = 0,
) -> Tuple[List[int], torch.Tensor]:
    """
    Sampling decoding with gradient, used for policy gradient.
    Returns:
      - token id list
      - sum of log probs (scalar tensor)
    """
    device = feats2d.device
    bos_id, eos_id = tok.bos_id, tok.eos_id

    seq = torch.full((1, 1), bos_id, device=device, dtype=torch.long)  # [1,1]
    log_prob_sum = torch.zeros(1, device=device)  # [1]

    for _ in range(max_len - 1):
        logits = model(
            feats2d,
            feats3d,
            seq,
            feat_pad_mask=feat_mask,
            obj_feats=obj_feats,
        )
        logits = logits[:, -1, :]  # [1, V]

        if topk and topk > 0:
            # top-k sampling
            topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)   # [1,K], [1,K]
            log_probs_topk = F.log_softmax(topk_vals, dim=-1)          # [1,K]
            probs_topk = log_probs_topk.exp()                          # [1,K]
            idx = torch.multinomial(probs_topk, num_samples=1)         # [1,1]
            next_id = topk_idx.gather(1, idx)                          # [1,1]
            log_prob = log_probs_topk.gather(1, idx)                   # [1,1]
        else:
            # full-vocab sampling
            log_probs = F.log_softmax(logits, dim=-1)                  # [1,V]
            probs = log_probs.exp()                                    # [1,V]
            next_id = torch.multinomial(probs, num_samples=1)          # [1,1]
            log_prob = log_probs.gather(1, next_id)                    # [1,1]

        seq = torch.cat([seq, next_id], dim=1)                         # [1, t+1]
        log_prob_sum = log_prob_sum + log_prob.squeeze()

        if next_id.item() == eos_id:
            break

    return seq[0].cpu().tolist(), log_prob_sum.squeeze()


# ---------------------------------------------------------
# collate for SCST (reuse train collate_fn + keep video_id)
# ---------------------------------------------------------
def collate_scst(batch, pad_id):
    """
    Reuse train collate_fn for padding, and keep video_id.
    """
    (
        feats2d,
        feats3d,
        obj_feats,
        feat_mask,
        y_in,
        y_out,
        tgt_mask,
        raw_caps,   # we ignore raw_caps here
    ) = train_collate_fn(batch, pad_id)
    video_ids = [b["video_id"] for b in batch]
    return feats2d, feats3d, obj_feats, feat_mask, y_in, y_out, tgt_mask, video_ids


# ---------------------------------------------------------
# main
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SCST (CIDEr-only) for MSRVTT")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--checkpoint", required=True, help="XE checkpoint to start from")
    ap.add_argument("--output_dir", required=True, help="Output dir for SCST checkpoints")
    ap.add_argument("--epochs", type=int, default=3, help="Number of SCST epochs")
    ap.add_argument("--batch_size", type=int, default=8, help="Batch size for SCST")
    ap.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Learning rate for SCST (smaller than XE)",
    )
    ap.add_argument(
        "--alpha_xe",
        type=float,
        default=1.0,
        help="Weight for XE loss in mixed objective",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Top-k sampling for SCST (0 = full vocab)",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SCST] Using device: {device}")

    # tokenizer
    tok = BPETokenizer(
        cfg["tokenizer"]["model_path"],
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"],
        cfg["tokenizer"]["pad_token"],
    )
    pad_id = tok.pad_id

    # feature roots
    def split_root(name: str) -> str:
        roots = cfg.get("features_root")
        if roots and isinstance(roots, dict):
            return roots.get(name, cfg.get("features_dir"))
        return os.path.join(cfg["features_dir"], name)

    # dataset / dataloader
    train_ds = MSRVTTFeatures(
        split_json=cfg["splits"]["train"],
        features_root=split_root("train"),
        tokenizer=tok,
        max_frames=cfg["max_frames"],
        use_3d=cfg.get("use_3d", True),
        use_obj=cfg.get("use_obj", True),
        caption_sampling="random",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=lambda b: collate_scst(b, pad_id),
    )

    # -----------------------------------------------------
    # Load XE checkpoint (to read align_dim and weights)
    # -----------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # If align_proj exists, read its dim; else use d_model
    align_dim = cfg["model"]["d_model"]
    if "align_proj.weight" in state:
        align_dim = state["align_proj.weight"].shape[1]
    print(f"[SCST] Using align_dim={align_dim}")

    # model (same arch as train/eval)
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
        use_3d_stream=cfg.get("use_3d_stream", True),
    ).to(device)

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[SCST] Loaded XE checkpoint from {args.checkpoint}")
    if missing:
        print(f"[SCST] Missing keys when loading checkpoint (OK if align-related only): {missing}")
    if unexpected:
        print(f"[SCST] Unexpected keys when loading checkpoint: {unexpected}")

    # -----------------------------------------------------
    # Freeze encoder / visual parts; train decoder only
    # -----------------------------------------------------
    for p in model.parameters():
        p.requires_grad = False
    for p in model.decoder.parameters():
        p.requires_grad = True

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SCST] Model params: {num_params:,} (trainable in SCST: {trainable_params:,})")

    model.train()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        betas=(0.9, 0.98),
        weight_decay=0.0,
    )

    # CIDEr references + global DF
    refs = load_multi_refs(cfg["msrvtt_raw_captions"])
    cider_df, cider_ref_len = build_cider_df(refs, n=4)

    # Check reference coverage on train split
    unique_vids = {_normalize_vid_key(it["video_id"]) for it in train_ds.items}
    hits = sum(1 for v in unique_vids if v in refs)
    if unique_vids:
        print(
            f"[SCST] Reference coverage: {hits}/{len(unique_vids)} unique videos "
            f"({100.0 * hits / len(unique_vids):.1f}%)"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    max_len = cfg["decoder"]["max_len"]
    grad_clip = cfg["optim"].get("grad_clip", 1.0)

    best_loss = float("inf")
    best_path = os.path.join(args.output_dir, "best_scst.pt")

    for epoch in range(args.epochs):
        model.train()
        epoch_rl_loss = 0.0
        epoch_xe_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_dl, desc=f"SCST Epoch {epoch+1}/{args.epochs}")
        for feats2d, feats3d, obj_feats, feat_mask, y_in, y_out, tgt_mask, video_ids in pbar:
            feats2d = feats2d.to(device)
            feats3d = feats3d.to(device)
            feat_mask = feat_mask.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)
            tgt_mask = tgt_mask.to(device)
            obj_feats = obj_feats.to(device) if obj_feats is not None else None

            B = feats2d.size(0)

            rewards_sample = []
            rewards_greedy = []
            log_probs = []

            # RL: per-sample decoding and CIDEr reward
            for b in range(B):
                vid_raw = video_ids[b]
                vid = _normalize_vid_key(vid_raw)
                refs_vid = refs.get(vid, [""])

                feats2d_b = feats2d[b:b + 1]
                feats3d_b = feats3d[b:b + 1]
                feat_mask_b = feat_mask[b:b + 1]
                obj_feats_b = obj_feats[b:b + 1] if obj_feats is not None else None

                # baseline: greedy decoding
                greedy_ids = greedy_decode(
                    model,
                    feats2d_b,
                    feats3d_b,
                    obj_feats_b,
                    feat_mask_b,
                    tok,
                    max_len=max_len,
                )

                # sampled caption (with grad)
                sample_ids, log_prob_sum = sample_decode(
                    model,
                    feats2d_b,
                    feats3d_b,
                    obj_feats_b,
                    feat_mask_b,
                    tok,
                    max_len=max_len,
                    topk=args.topk,
                )

                greedy_text = decode_tokens(greedy_ids, tok)
                sample_text = decode_tokens(sample_ids, tok)

                # CIDEr-only rewards
                r_g = compute_cider_reward_single(
                    greedy_text, refs_vid, cider_df, cider_ref_len, n=4
                )
                r_s = compute_cider_reward_single(
                    sample_text, refs_vid, cider_df, cider_ref_len, n=4
                )

                rewards_greedy.append(r_g)
                rewards_sample.append(r_s)
                log_probs.append(log_prob_sum)

                if epoch == 0 and num_batches == 0 and b < 2:
                    print(
                        f"[DEBUG] vid={vid}, "
                        f"CIDEr_s={r_s:.3f}, CIDEr_g={r_g:.3f}, "
                        f"adv={r_s - r_g:.3f}, logp={log_prob_sum.item():.3f}"
                    )

            # convert rewards and log probs to tensors
            rewards_sample_t = torch.tensor(rewards_sample, dtype=torch.float32, device=device)
            rewards_greedy_t = torch.tensor(rewards_greedy, dtype=torch.float32, device=device)
            advantages = rewards_sample_t - rewards_greedy_t

            # normalize advantages to reduce variance
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            log_probs_t = torch.stack(log_probs)  # [B]

            rl_loss = -(advantages * log_probs_t).mean()

            # XE loss for extra stability
            logits = model(
                feats2d,
                feats3d,
                y_in,
                feat_pad_mask=feat_mask,
                tgt_pad_mask=tgt_mask,
                obj_feats=obj_feats,
            )
            xe = xe_loss(logits, y_out, pad_id)

            total_loss = rl_loss + args.alpha_xe * xe

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_rl_loss += rl_loss.item()
            epoch_xe_loss += xe.item()
            num_batches += 1

            pbar.set_postfix(
                {
                    "rl_loss": f"{rl_loss.item():.4f}",
                    "xe_loss": f"{xe.item():.4f}",
                }
            )

        avg_rl_loss = epoch_rl_loss / max(num_batches, 1)
        avg_xe_loss = epoch_xe_loss / max(num_batches, 1)
        avg_total_loss = avg_rl_loss + args.alpha_xe * avg_xe_loss

        print(
            f"[SCST] Epoch {epoch+1} - "
            f"Avg RL Loss: {avg_rl_loss:.4f}, "
            f"Avg XE Loss: {avg_xe_loss:.4f}, "
            f"Avg Total: {avg_total_loss:.4f}"
        )

        ckpt_path = os.path.join(args.output_dir, f"scst_epoch{epoch+1}.pt")
        torch.save({"model": model.state_dict()}, ckpt_path)
        print(f"[SCST] Saved checkpoint: {ckpt_path}")

        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            torch.save({"model": model.state_dict()}, best_path)
            print(f"[SCST] Updated best_scst at {best_path} (avg_total_loss={avg_total_loss:.4f})")


if __name__ == "__main__":
    main()
