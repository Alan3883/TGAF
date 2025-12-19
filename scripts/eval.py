import argparse
import os
import pickle
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPModel

from src.models.model import VideoCaptionModel
from src.utils.tokenizer import BPETokenizer
from src.data.datasets import MSRVTTFeatures
from src.data.collate import collate_fn

# ---------------------------------------------------------------------
# COCO-style metrics
# ---------------------------------------------------------------------
try:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    EVAL_COCO_AVAILABLE = True
except ImportError:
    EVAL_COCO_AVAILABLE = False
    print("Warning: COCO metrics not available. Install pycocoevalcap for BLEU/METEOR/ROUGE/CIDEr")

# ---------------------------------------------------------------------
# SacreBLEU fallback
# ---------------------------------------------------------------------
try:
    from sacrebleu import corpus_bleu

    SACRE_AVAILABLE = True
except ImportError:
    SACRE_AVAILABLE = False
    print("Warning: sacrebleu not available. Install sacrebleu for BLEU fallback")


# ---------------------------------------------------------------------
# Beam search decoding (now returns top-K candidates for reranking)
# ---------------------------------------------------------------------
@torch.no_grad()
def beam_search_decode(
    model,
    feats2d: torch.Tensor | None,
    feats3d: torch.Tensor | None,
    obj_feats: torch.Tensor | None,
    feat_mask: torch.Tensor,
    tok,
    max_len: int = 40,
    beam_width: int = 5,
    length_alpha: float = 0.7,
    repetition_penalty: float = 1.2,
    min_len: int = 5,
    num_return_sequences: int = 1,
):
    """
    Beam search with:
      - length normalization
      - simple repetition penalty
      - minimum length constraint

    Returns:
      list of (norm_lm_score, log_prob, seq_tensor) for top-K beams
      where:
        norm_lm_score = log_prob / length_penalty(L)
        seq_tensor    = [1, L] tensor (B=1)
    """
    device = feat_mask.device
    bos_id, eos_id = tok.bos_id, tok.eos_id

    # (cum_logprob, seq[tensor])
    beams: list[tuple[float, torch.Tensor]] = [
        (0.0, torch.tensor([[bos_id]], device=device, dtype=torch.long))
    ]

    def length_penalty(length: int) -> float:
        if length_alpha <= 0.0:
            return 1.0
        return ((5.0 + length) ** length_alpha) / ((5.0 + 1.0) ** length_alpha)

    def scored_beams(candidates):
        # sort candidates by normalized log prob for pruning
        scored = []
        for log_p, seq in candidates:
            L = seq.size(1)
            lp = length_penalty(L)
            scored.append((log_p / lp, log_p, seq))
        scored.sort(key=lambda x: x[0], reverse=True)
        # keep only log_p, seq for next step
        return [(log_p, seq) for _, log_p, seq in scored]

    for _ in range(max_len - 1):
        new_beams: list[tuple[float, torch.Tensor]] = []

        for log_p, seq in beams:
            last_token = seq[0, -1].item()

            # already ended
            if last_token == eos_id:
                new_beams.append((log_p, seq))
                continue

            logits = model(
                feats2d,
                feats3d,
                seq,
                obj_feats=obj_feats,
                feat_pad_mask=feat_mask,
            )
            next_logits = logits[:, -1, :].squeeze(0)  # [V]

            # 1) minimum length constraint
            if seq.size(1) < min_len:
                next_logits[eos_id] = -float("inf")

            # 2) block immediate repetition
            next_logits[last_token] = -float("inf")

            # 3) repetition penalty
            if repetition_penalty is not None and repetition_penalty > 1.0:
                generated = set(seq[0].tolist())
                for token_id in generated:
                    if token_id in {bos_id, eos_id, tok.pad_id}:
                        continue
                    logit = next_logits[token_id]
                    if logit > 0:
                        next_logits[token_id] = logit / repetition_penalty
                    else:
                        next_logits[token_id] = logit * repetition_penalty

            log_probs = torch.log_softmax(next_logits, dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, beam_width, dim=-1)

            for k in range(beam_width):
                token_id = topk_idx[k].unsqueeze(0).unsqueeze(0)  # [1,1]
                candidate_seq = torch.cat([seq, token_id], dim=1)  # [1, L+1]
                candidate_log_p = log_p + topk_logp[k].item()
                new_beams.append((candidate_log_p, candidate_seq))

        beams = scored_beams(new_beams)[:beam_width]

        if all(seq[0, -1].item() == eos_id for _, seq in beams):
            break

    # Final sort by normalized LM score
    def length_penalty_final(seq: torch.Tensor, lp_alpha: float) -> float:
        L = seq.size(1)
        if lp_alpha <= 0.0:
            return 1.0
        return ((5.0 + L) ** lp_alpha) / ((5.0 + 1.0) ** lp_alpha)

    beams = sorted(
        beams,
        key=lambda x: x[0] / length_penalty_final(x[1], length_alpha),
        reverse=True,
    )

    candidates: list[tuple[float, float, torch.Tensor]] = []
    for log_p, seq in beams[:num_return_sequences]:
        lp = length_penalty_final(seq, length_alpha)
        norm_lm = log_p / lp
        candidates.append((norm_lm, log_p, seq))
    return candidates


def decode_tokens(token_ids, tok):
    """Decode token IDs to plain text (drop BOS/EOS/PAD)."""
    special_ids = {tok.bos_id, tok.eos_id, tok.pad_id}
    clean_ids = [tid for tid in token_ids if tid not in special_ids]
    return tok.decode(clean_ids)


# ---------------------------------------------------------------------
# Multi-reference loader
# ---------------------------------------------------------------------
def _normalize_vid_key(key):
    """
    Normalize video key to match dataset IDs:
      - int -> "videoXXXX"
      - str -> strip ".mp4"
    """
    if isinstance(key, int):
        vid = f"video{key}"
    else:
        vid = str(key)
    if vid.endswith(".mp4"):
        vid = vid[:-4]
    return vid


def load_multi_reference_captions(pkl_path: str):
    """
    Load raw-captions.pkl with structure:
        {
          "video2960": [
            ["a","cartoon","animals",...],
            ["a","cartoon","character",...],
            ...
          ],
          ...
        }
    Return:
        refs[video_id] = ["a cartoon animals ...", "a cartoon character ...", ...]
    """
    if not os.path.exists(pkl_path):
        print(f"Multi-ref file not found at {pkl_path} (will fall back to 1 caption per video).")
        return {}

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print(f"Unexpected raw caption format (type={type(data)}), expect dict.")
        return {}

    refs: dict[str, list[str]] = {}

    for vid_key, sent_list in data.items():
        vid = _normalize_vid_key(vid_key)
        if not isinstance(sent_list, list):
            continue

        caps: list[str] = []
        for tokens in sent_list:
            if isinstance(tokens, list):
                words = [t for t in tokens if isinstance(t, str)]
                if not words:
                    continue
                s = " ".join(words).strip()
                if s:
                    caps.append(s)

        if caps:
            refs[vid] = caps

    print(f"Loaded multi-reference captions for {len(refs)} videos from {pkl_path}")
    return refs


# ---------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------
def evaluate_coco_metrics(predictions, references):
    """
    COCO-style metrics (BLEU@1-4, METEOR, ROUGE_L, CIDEr).
    predictions: {vid: "hyp"}
    references: {vid: ["ref1", ..., "refK"]}
    """
    if not EVAL_COCO_AVAILABLE:
        return {}

    gts = {vid: refs for vid, refs in references.items()}
    res = {vid: [pred] for vid, pred in predictions.items()}

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gts, res)
        if isinstance(method, list):
            for sc, m in zip(score, method):
                scores[m] = sc
        else:
            scores[method] = score
    return scores


def evaluate_sacrebleu(pred_texts, ref_texts):
    """
    Fallback corpus BLEU-4 using sacrebleu.
    pred_texts: [hyp1, hyp2, ...]
    ref_texts:  [[ref1_1,...], [ref2_1,...], ...]
    """
    if not SACRE_AVAILABLE or not pred_texts:
        return None

    if not ref_texts:
        ref_streams = [[""] * len(pred_texts)]
    elif isinstance(ref_texts[0], list):
        max_refs = max(len(r) for r in ref_texts if r) if ref_texts else 1
        ref_streams = [[] for _ in range(max_refs)]
        for refs in ref_texts:
            if not refs:
                refs = [""]
            if len(refs) < max_refs:
                refs = refs + [refs[0]] * (max_refs - len(refs))
            for i, r in enumerate(refs):
                ref_streams[i].append(r)
    else:
        ref_streams = [ref_texts]

    bleu = corpus_bleu(pred_texts, ref_streams)
    return bleu.score


# ---------------------------------------------------------------------
# CLIP-guided reranking helpers
# ---------------------------------------------------------------------
@torch.no_grad()
def pooled_video_repr(enc_states: torch.Tensor, feat_mask: torch.Tensor | None):
    """
    Temporal mean pooling with mask to get video-level embedding.

    enc_states: [B, T, D]
    feat_mask:  [B, T] (True = padding) or None
    """
    if feat_mask is not None:
        valid = (~feat_mask.bool()).float()           # [B, T]
        lengths = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        video_repr = (enc_states * valid.unsqueeze(-1)).sum(dim=1) / lengths
    else:
        video_repr = enc_states.mean(dim=1)
    return video_repr  # [B, D]


@torch.no_grad()
def clip_rerank_candidates(
    candidates,             # list of (norm_lm, log_p, seq_tensor)
    feats2d,
    feats3d,
    obj_feats,
    feat_mask,
    tok,
    model,
    clip_tokenizer,
    clip_model,
    clip_lambda: float,
    device,
):
    """
    CLIP-guided reranking of beam candidates.

    score = norm_lm + clip_lambda * sim_clip
    where sim_clip is cosine similarity between:
      - video embedding (model.encode + align_proj)
      - CLIP text embedding for candidate caption
    """
    # 1) Video embedding in CLIP-aligned space
    enc_states, _, _ = model.encode(
        feats2d,
        feats3d,
        feat_pad_mask=feat_mask,
        obj_feats=obj_feats,
        return_obj_align=False,
    )  # [1, T, D_model]

    video_repr = pooled_video_repr(enc_states, feat_mask)        # [1, D_model]
    video_proj = model.align_proj(video_repr)                    # [1, D_clip]
    video_norm = F.normalize(video_proj, dim=-1)                 # [1, D_clip]

    best_score = None
    best_seq_ids = None

    for norm_lm, log_p, seq in candidates:
        token_ids = seq[0].cpu().tolist()
        text = decode_tokens(token_ids, tok)
        if not text.strip():
            sim = 0.0
        else:
            inputs = clip_tokenizer(
                [text],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            text_features = clip_model.get_text_features(**inputs)  # [1, D_clip]
            text_norm = F.normalize(text_features, dim=-1)
            sim = (video_norm * text_norm).sum(dim=-1).item()

        total_score = norm_lm + clip_lambda * sim
        if best_score is None or total_score > best_score:
            best_score = total_score
            best_seq_ids = token_ids

    return best_seq_ids


# ---------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------
def main(cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tokenizer
    tok = BPETokenizer(
        cfg["tokenizer"]["model_path"],
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"],
        cfg["tokenizer"]["pad_token"],
    )
    pad_id = tok.pad_id

    # Feature root helper
    def split_root(name: str) -> str:
        roots = cfg.get("features_root")
        if roots and isinstance(roots, dict):
            return roots.get(name, cfg.get("features_dir"))
        return os.path.join(cfg["features_dir"], name)

    # Test dataset
    test_ds = MSRVTTFeatures(
        split_json=cfg["splits"]["test"],
        features_root=split_root("test"),
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

    # Multi-reference captions
    raw_caps_path = cfg.get("msrvtt_raw_captions", "data/msrvtt/raw/raw-captions.pkl")
    multi_refs = load_multi_reference_captions(raw_caps_path)

    # Checkpoint + align_dim (for align_proj)
    checkpoint_path = cfg.get("checkpoint", os.path.join(cfg["output_dir"], "last.pt"))
    ckpt = None
    align_dim = cfg["model"]["d_model"]

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        if "align_proj.weight" in state:
            align_dim = state["align_proj.weight"].shape[1]
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: checkpoint not found at {checkpoint_path}, using random weights")

    # Model (align_dim must match training)
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

    if ckpt is not None:
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)

    model.eval()

    # -----------------------------------------------------------------
    # Decoding hyperparameters (config + CLI override)
    # -----------------------------------------------------------------
    dec_cfg = cfg.get("decoding", {})
    if args.beam_width is not None:
        beam_width = args.beam_width
    else:
        beam_width = dec_cfg.get("beam_width", cfg.get("beam_width", 5))

    if args.length_alpha is not None:
        length_alpha = args.length_alpha
    else:
        length_alpha = dec_cfg.get("length_alpha", 0.7)

    if args.repetition_penalty is not None:
        repetition_penalty = args.repetition_penalty
    else:
        repetition_penalty = dec_cfg.get("repetition_penalty", 1.2)

    if args.min_len is not None:
        min_len = args.min_len
    else:
        min_len = dec_cfg.get("min_len", 5)

    beam_rerank_k = args.beam_rerank_k

    print(
        f"[Decoding] beam_width={beam_width}, length_alpha={length_alpha}, "
        f"repetition_penalty={repetition_penalty}, min_len={min_len}, "
        f"beam_rerank_k={beam_rerank_k}"
    )

    # -----------------------------------------------------------------
    # CLIP text encoder for reranking (optional)
    # -----------------------------------------------------------------
    clip_tokenizer = None
    clip_model = None
    clip_lambda = args.clip_lambda
    use_clip_rerank = args.clip_rerank

    if use_clip_rerank:
        clip_cfg = cfg.get("clip", {})
        clip_model_name = (
            args.clip_model_name
            or clip_cfg.get("text_encoder")
            or "openai/clip-vit-base-patch32"
        )
        print(
            f"[CLIP rerank] Enabled. Model: {clip_model_name}, lambda={clip_lambda}"
        )
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_model.to(device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

    predictions = {}
    references = {}
    ordered_ids = []

    print("Generating captions...")
    for idx, batch in enumerate(tqdm(test_dl)):
        # collate_fn returns 8 elements (raw_caps last, unused here)
        feats2d, feats3d, obj_feats, feat_mask, _, _, _, _ = batch
        feats2d = feats2d.to(device)
        feats3d = feats3d.to(device)
        obj_feats = obj_feats.to(device) if obj_feats is not None else None
        feat_mask = feat_mask.to(device)

        # video_id from dataset, e.g. "video2960" or "video2960.mp4"
        video_id = (
            test_ds.items[idx]["video_id"] if hasattr(test_ds, "items") else f"video_{idx}"
        )
        norm_vid = _normalize_vid_key(video_id)

        # 1) Beam search -> get top-K candidates from LM
        candidates = beam_search_decode(
            model,
            feats2d,
            feats3d,
            obj_feats,
            feat_mask,
            tok,
            max_len=cfg["decoder"]["max_len"],
            beam_width=beam_width,
            length_alpha=length_alpha,
            repetition_penalty=repetition_penalty,
            min_len=min_len,
            num_return_sequences=beam_rerank_k,
        )

        # 2) CLIP-guided reranking (optional)
        if use_clip_rerank and clip_tokenizer is not None and clip_model is not None:
            token_ids = clip_rerank_candidates(
                candidates,
                feats2d,
                feats3d,
                obj_feats,
                feat_mask,
                tok,
                model,
                clip_tokenizer,
                clip_model,
                clip_lambda=clip_lambda,
                device=device,
            )
        else:
            # Fallback: purely LM-based best beam
            norm_lm, log_p, seq = candidates[0]
            token_ids = seq[0].cpu().tolist()

        pred_text = decode_tokens(token_ids, tok)

        # References: prefer multi-refs, fallback to split json
        if norm_vid in multi_refs and multi_refs[norm_vid]:
            ref_captions = multi_refs[norm_vid]
        else:
            raw_ref = test_ds.items[idx]["caption"] if hasattr(test_ds, "items") else ""
            if isinstance(raw_ref, list):
                ref_captions = [
                    c.strip() for c in raw_ref if isinstance(c, str) and c.strip()
                ]
            else:
                ref_captions = (
                    [raw_ref.strip()]
                    if isinstance(raw_ref, str) and raw_ref.strip()
                    else [""]
                )

        predictions[video_id] = pred_text
        references[video_id] = ref_captions
        ordered_ids.append(video_id)

    # Metrics
    print("\nEvaluating...")
    scores = evaluate_coco_metrics(predictions, references)

    # Optional sacreBLEU
    if "Bleu_1" not in scores and SACRE_AVAILABLE:
        hyp_list = [predictions[vid] for vid in ordered_ids]
        ref_list = [references[vid] for vid in ordered_ids]
        sacre_bleu = evaluate_sacrebleu(hyp_list, ref_list)
        if sacre_bleu is not None:
            scores["sacreBLEU"] = sacre_bleu
    if not scores:
        scores["warning"] = "No evaluation metrics available; install pycocoevalcap or sacrebleu."

    print("\nEvaluation Results:")
    for metric, score in scores.items():
        if isinstance(score, (int, float)):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")

    # Save predictions + refs
    output_file = os.path.join(cfg["output_dir"], "predictions.txt")
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for vid, pred in predictions.items():
            refs = references[vid] if references[vid] else [""]
            f.write(f"{vid}\n")
            f.write(f"Pred: {pred}\n")
            f.write(f"Refs: {refs}\n\n")
    print(f"\nPredictions saved to {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/msrvtt.yaml", help="Config file path")
    ap.add_argument("--checkpoint", default=None, help="Path to model checkpoint")

    # Decoding hyperparams (can override config)
    ap.add_argument("--beam_width", type=int, default=None, help="Beam width")
    ap.add_argument(
        "--length_alpha",
        type=float,
        default=None,
        help="Length penalty alpha for beam search",
    )
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty (>1.0)",
    )
    ap.add_argument(
        "--min_len",
        type=int,
        default=None,
        help="Minimum caption length (steps before EOS allowed)",
    )

    # CLIP reranking
    ap.add_argument(
        "--clip_rerank",
        action="store_true",
        help="Enable CLIP-guided beam reranking",
    )
    ap.add_argument(
        "--clip_lambda",
        type=float,
        default=1.0,
        help="Weight for CLIP similarity in reranking score",
    )
    ap.add_argument(
        "--clip_model_name",
        type=str,
        default=None,
        help="CLIP text encoder name (default: config['clip']['text_encoder'] or openai/clip-vit-base-patch32)",
    )
    ap.add_argument(
        "--beam_rerank_k",
        type=int,
        default=5,
        help="How many beams to rerank with CLIP",
    )

    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint

    main(cfg, args)
