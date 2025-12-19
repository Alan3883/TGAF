import argparse
import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from transformers import CLIPTokenizer, CLIPModel

from src.utils.tokenizer import BPETokenizer
from src.data.datasets import MSRVTTFeatures
from src.data.collate import collate_fn
from src.models.model import VideoCaptionModel


def xe_loss(logits, y_out, pad_id):
    """
    Caption cross-entropy loss with label smoothing.

    logits: [B, L, V]
    y_out:  [B, L]
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y_out.view(-1),
        ignore_index=pad_id,
        label_smoothing=0.1,
    )


def object_align_loss(enc_states, obj_states, feat_mask):
    """
    Object-aware encoder alignment loss.

    enc_states: [B, T, D]
    obj_states: [B, T, D]
    feat_mask:  [B, T] (True = padding)
    """
    if obj_states is None:
        return enc_states.new_tensor(0.0)

    valid = ~feat_mask.bool()  # [B, T]
    if not valid.any():
        return enc_states.new_tensor(0.0)

    enc_flat = enc_states[valid]  # [N, D]
    obj_flat = obj_states[valid]  # [N, D]

    sim = torch.nn.functional.cosine_similarity(enc_flat, obj_flat, dim=-1)  # [N]
    loss = 1.0 - sim.mean()
    return loss


def pooled_video_repr(enc_states, feat_mask):
    """
    Temporal pooling of encoder states to get a video-level embedding.

    enc_states: [B, T, D]
    feat_mask:  [B, T] (True = padding)
    return: [B, D]
    """
    if feat_mask is not None:
        valid = (~feat_mask).float()          # [B, T]
        lengths = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        video_repr = (enc_states * valid.unsqueeze(-1)).sum(dim=1) / lengths
    else:
        video_repr = enc_states.mean(dim=1)
    return video_repr


def clip_contrastive_loss(
    video_repr,
    raw_caps,
    clip_tokenizer,
    clip_model,
    device,
    temperature=0.07,
):
    """
    InfoNCE contrastive loss between video_repr and CLIP text features.

    video_repr: [B, D_v] (already projected to CLIP dim)
    raw_caps: list[str] length B
    """
    if len(raw_caps) == 0:
        return video_repr.new_tensor(0.0)

    with torch.no_grad():
        inputs = clip_tokenizer(
            list(raw_caps),
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        text_features = clip_model.get_text_features(**inputs)  # [B, D_clip]

    B = video_repr.size(0)
    v = torch.nn.functional.normalize(video_repr, dim=-1)
    t = torch.nn.functional.normalize(text_features, dim=-1)

    logits_vt = (v @ t.t()) / temperature      # [B, B]
    logits_tv = logits_vt.t()                  # [B, B]
    labels = torch.arange(B, device=device)

    loss_v2t = torch.nn.functional.cross_entropy(logits_vt, labels)
    loss_t2v = torch.nn.functional.cross_entropy(logits_tv, labels)
    loss = 0.5 * (loss_v2t + loss_t2v)
    return loss


@torch.no_grad()
def validate(
    model,
    val_dl,
    pad_id,
    device,
    lambda_obj=0.0,
    lambda_clip=0.0,
    clip_tokenizer=None,
    clip_model=None,
    temperature=0.07,
):
    """
    Validation: XE + optional object alignment + CLIP contrastive loss.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_dl:
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
        obj_feats = obj_feats.to(device) if obj_feats is not None else None
        feat_mask = feat_mask.to(device)
        y_in = y_in.to(device)
        y_out = y_out.to(device)
        tgt_mask = tgt_mask.to(device)

        logits, enc_states, obj_states, _ = model(
            feats2d,
            feats3d,
            y_in,
            obj_feats=obj_feats,
            feat_pad_mask=feat_mask,
            tgt_pad_mask=tgt_mask,
            return_enc_states=True,
        )

        cap_loss = xe_loss(logits, y_out, pad_id)
        obj_loss = object_align_loss(enc_states, obj_states, feat_mask)

        if (
            lambda_clip > 0.0
            and clip_model is not None
            and clip_tokenizer is not None
            and hasattr(model, "align_proj")
        ):
            enc_video = pooled_video_repr(enc_states, feat_mask)  # [B, D_model]
            enc_video_proj = model.align_proj(enc_video)          # [B, D_clip]
            clip_loss = clip_contrastive_loss(
                enc_video_proj,
                raw_caps,
                clip_tokenizer,
                clip_model,
                device,
                temperature=temperature,
            )
        else:
            clip_loss = enc_states.new_tensor(0.0)

        loss = cap_loss + lambda_obj * obj_loss + lambda_clip * clip_loss

        total_loss += loss.item()
        num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------
    # Loss weights
    # -------------------------
    loss_cfg = cfg.get("loss", {})
    lambda_obj = float(loss_cfg.get("lambda_obj", 0.0))
    # use lambda_clip; fall back to old lambda_align if present
    lambda_clip = float(loss_cfg.get("lambda_clip", loss_cfg.get("lambda_align", 0.0)))

    # -------------------------
    # Backbone info (just for logging)
    # -------------------------
    features_dir = cfg.get("features_dir", "")
    feat2d_dim = cfg.get("feature_dim_2d", None)
    backbone_name = cfg.get("backbone", None)
    if backbone_name is None:
        low = str(features_dir).lower()
        if "vit32" in low or "vit" in low or "clip" in low:
            backbone_name = "CLIP-ViT"
        elif "resnet" in low:
            backbone_name = "ResNet-50"
        else:
            backbone_name = "unknown"
    print(
        f"[Backbone] 2D features from '{features_dir}' "
        f"(backbone={backbone_name}, dim={feat2d_dim})"
    )

    # -------------------------
    # Output & Logging
    # -------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    log_dir = os.path.join(cfg["output_dir"], "logs")
    writer = SummaryWriter(log_dir)

    # -------------------------
    # Tokenizers
    # -------------------------
    tok = BPETokenizer(
        cfg["tokenizer"]["model_path"],
        cfg["tokenizer"]["bos_token"],
        cfg["tokenizer"]["eos_token"],
        cfg["tokenizer"]["pad_token"],
    )
    pad_id = tok.pad_id

    # -------------------------
    # Basic sanity checks on loss config
    # -------------------------
    use_obj = cfg.get("use_obj", True)
    if lambda_obj > 0.0 and (not use_obj or cfg.get("feature_dim_obj", 0) <= 0):
        print(
            "[WARN] loss.lambda_obj > 0 but use_obj=False or feature_dim_obj<=0; "
            "setting lambda_obj = 0.0."
        )
        lambda_obj = 0.0

    # -------------------------
    # CLIP text encoder (optional)
    # -------------------------
    clip_cfg = cfg.get("clip", {})
    # if config doesn't specify enabled, fall back to "lambda_clip > 0"
    clip_enabled = bool(clip_cfg.get("enabled", lambda_clip > 0.0))
    clip_temperature = float(clip_cfg.get("temperature", 0.07))

    if clip_enabled and lambda_clip > 0.0:
        clip_model_name = clip_cfg.get("text_encoder", "openai/clip-vit-base-patch32")
        print(
            f"[CLIP] Loading TEXT encoder (not the video backbone): {clip_model_name}"
        )
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        clip_model = CLIPModel.from_pretrained(clip_model_name)
        clip_model.to(device)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        clip_dim = clip_model.config.projection_dim
    else:
        if lambda_clip > 0.0 and not clip_enabled:
            print(
                "[WARN] loss.lambda_clip > 0 but clip.enabled is False; "
                "CLIP contrastive loss will be skipped."
            )
        clip_enabled = False
        clip_tokenizer = None
        clip_model = None
        clip_dim = None
        lambda_clip = 0.0  # make sure it's really off

    print(
        f"[Loss config] lambda_obj={lambda_obj:.3f}, "
        f"lambda_clip={lambda_clip:.3f}, clip_enabled={clip_enabled}"
    )

    # -------------------------
    # Datasets
    # -------------------------
    def split_root(name: str) -> str:
        roots = cfg.get("features_root")
        if roots and isinstance(roots, dict):
            return roots.get(name, cfg.get("features_dir"))
        # fallback: features_dir/<split>
        return os.path.join(cfg["features_dir"], name)

    train_ds = MSRVTTFeatures(
        split_json=cfg["splits"]["train"],
        features_root=split_root("train"),
        tokenizer=tok,
        max_frames=cfg["max_frames"],
        use_3d=cfg.get("use_3d", True),
        use_obj=use_obj,
        caption_sampling="random",
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    val_dl = None
    if "val" in cfg.get("splits", {}):
        val_ds = MSRVTTFeatures(
            split_json=cfg["splits"]["val"],
            features_root=split_root("val"),
            tokenizer=tok,
            max_frames=cfg["max_frames"],
            use_3d=cfg.get("use_3d", True),
            use_obj=use_obj,
            caption_sampling="first",
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            collate_fn=lambda b: collate_fn(b, pad_id),
        )

    # -------------------------
    # Model
    # -------------------------
    model_kwargs = dict(
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
        use_3d_stream=cfg.get("use_3d_stream", True),
    )

    # Only pass align_dim when we really use CLIP alignment
    if clip_enabled and lambda_clip > 0.0 and clip_dim is not None:
        model = VideoCaptionModel(align_dim=clip_dim, **model_kwargs).to(device)
    else:
        model = VideoCaptionModel(**model_kwargs).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} (trainable: {trainable_params:,})")

    # -------------------------
    # Optimizer & Scheduler
    # -------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )

    scheduler = None
    if cfg.get("scheduler"):
        if cfg["scheduler"]["type"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg["epochs"]
            )

    # -------------------------
    # Training loop
    # -------------------------
    global_step = 0
    best_val_loss = float("inf") if val_dl is not None else None
    best_path = os.path.join(cfg["output_dir"], "best.pt")

    print(f"Starting training for {cfg['epochs']} epochs...")
    for epoch in range(cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_cap_loss = 0.0
        epoch_obj_loss = 0.0
        epoch_clip_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        for batch in pbar:
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
            obj_feats = obj_feats.to(device) if obj_feats is not None else None
            feat_mask = feat_mask.to(device)
            y_in = y_in.to(device)
            y_out = y_out.to(device)
            tgt_mask = tgt_mask.to(device)

            logits, enc_states, obj_states, _ = model(
                feats2d,
                feats3d,
                y_in,
                obj_feats=obj_feats,
                feat_pad_mask=feat_mask,
                tgt_pad_mask=tgt_mask,
                return_enc_states=True,
            )

            cap_loss = xe_loss(logits, y_out, pad_id)
            obj_loss = object_align_loss(enc_states, obj_states, feat_mask)

            if (
                lambda_clip > 0.0
                and clip_enabled
                and clip_model is not None
                and clip_tokenizer is not None
                and hasattr(model, "align_proj")
            ):
                enc_video = pooled_video_repr(enc_states, feat_mask)  # [B, D_model]
                enc_video_proj = model.align_proj(enc_video)          # [B, D_clip]
                clip_loss = clip_contrastive_loss(
                    enc_video_proj,
                    raw_caps,
                    clip_tokenizer,
                    clip_model,
                    device,
                    temperature=clip_temperature,
                )
            else:
                clip_loss = enc_states.new_tensor(0.0)

            loss = cap_loss + lambda_obj * obj_loss + lambda_clip * clip_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"]["grad_clip"])
            opt.step()

            epoch_loss += loss.item()
            epoch_cap_loss += cap_loss.item()
            epoch_obj_loss += obj_loss.item()
            epoch_clip_loss += clip_loss.item()
            num_batches += 1
            global_step += 1

            if global_step % cfg["log_every"] == 0:
                avg_loss = epoch_loss / num_batches
                avg_cap = epoch_cap_loss / num_batches
                avg_obj = epoch_obj_loss / num_batches
                avg_clip = epoch_clip_loss / num_batches
                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "cap": f"{avg_cap:.4f}",
                        "obj": f"{avg_obj:.4f}",
                        "clip": f"{avg_clip:.4f}",
                    }
                )
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/cap_loss", cap_loss.item(), global_step)
                writer.add_scalar("train/obj_loss", obj_loss.item(), global_step)
                writer.add_scalar("train/clip_loss", clip_loss.item(), global_step)
                writer.add_scalar("train/lr", opt.param_groups[0]["lr"], global_step)

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_cap_epoch = epoch_cap_loss / num_batches if num_batches > 0 else 0.0
        avg_obj_epoch = epoch_obj_loss / num_batches if num_batches > 0 else 0.0
        avg_clip_epoch = epoch_clip_loss / num_batches if num_batches > 0 else 0.0

        print(
            f"Epoch {epoch+1} - "
            f"Train Loss: {avg_epoch_loss:.4f} | "
            f"Cap: {avg_cap_epoch:.4f} | Obj: {avg_obj_epoch:.4f} | Clip: {avg_clip_epoch:.4f}"
        )
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        writer.add_scalar("train/epoch_cap_loss", avg_cap_epoch, epoch)
        writer.add_scalar("train/epoch_obj_loss", avg_obj_epoch, epoch)
        writer.add_scalar("train/epoch_clip_loss", avg_clip_epoch, epoch)

        # Validation
        if val_dl is not None:
            val_loss = validate(
                model,
                val_dl,
                pad_id,
                device,
                lambda_obj=lambda_obj,
                lambda_clip=lambda_clip,
                clip_tokenizer=clip_tokenizer,
                clip_model=clip_model,
                temperature=clip_temperature,
            )
            print(f"Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
            writer.add_scalar("val/loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "config": cfg,
                    },
                    best_path,
                )
                print(f"Saved best model (val_loss={val_loss:.4f}) to {best_path}")
        else:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_epoch_loss,
                    "config": cfg,
                },
                best_path,
            )
            print(f"Saved checkpoint as best.pt (no validation set) to {best_path}")

        # Save last checkpoint
        checkpoint_path = os.path.join(cfg["output_dir"], "last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "config": cfg,
            },
            checkpoint_path,
        )

        if scheduler is not None:
            scheduler.step()

    if best_val_loss is not None:
        print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    else:
        print("Training completed with no validation set; latest model saved to best.pt")
    writer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train video captioning model")
    ap.add_argument("--config", default="configs/msrvtt.yaml", help="Config file path")
    # NOTE: --resume is currently unused; if你以后想做断点续训，可以再扩展
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    main(cfg)
