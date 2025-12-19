# src/models/model.py

"""
Architecture:
  1. Object features -> FiLM-style modulation of 2D frame features
  2. Project 2D / 3D into d_model
  3. Two temporal streams:
       - 2D stream: Dilated TCN over 2D features
       - 3D stream: Dilated TCN over 3D features
  4. GMU-style fusion of 2D / 3D -> fused visual stream (for losses)
  5. Optional object temporal stream -> gated fusion into fused visual stream
  6. Transformer decoder:
       - cross-attention to 2D & 3D streams
       - per-token gate to choose 2D vs 3D context
"""

import torch
import torch.nn as nn
from .tcn import DilatedTCN
from .decoder import TransformerDecoder


class VideoCaptionModel(nn.Module):
    """
    Video captioning model with 2D / 3D / object feature fusion, dual TCN encoders,
    and a gated multi-stream Transformer decoder.
    """

    def __init__(
        self,
        vocab_size: int,
        feature_dim_2d: int,
        feature_dim_3d: int,
        feature_dim_obj: int = 0,
        d_model: int = 512,
        tcn_layers: int = 6,
        k: int = 3,
        dilations=None,
        dec_layers: int = 3,
        nhead: int = 8,
        ff: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
        align_dim: int | None = None,  # projection dim for CLIP space (if used)
        use_3d_stream: bool = True,    # NEW: enable / disable 3D temporal stream
    ):
        super().__init__()

        if feature_dim_2d is None or feature_dim_2d <= 0:
            raise ValueError("feature_dim_2d must be > 0")
        if feature_dim_3d is None or feature_dim_3d <= 0:
            raise ValueError("feature_dim_3d must be > 0")

        self.use_obj = feature_dim_obj is not None and feature_dim_obj > 0
        self.use_3d_stream = use_3d_stream

        # -------- Obj → 2D: FiLM-style modulation + object temporal stream --------
        if self.use_obj:
            # FiLM modulation: obj -> (gamma, beta) in 2D feature space
            self.obj_to_gamma_beta = nn.Linear(feature_dim_obj, 2 * feature_dim_2d)
            self.obj_ln = nn.LayerNorm(feature_dim_2d)

            # Obj-aligned projection for auxiliary loss (obj -> d_model)
            self.obj_align_proj = nn.Linear(feature_dim_obj, d_model)
            self.obj_align_ln = nn.LayerNorm(d_model)

            # Temporal encoder for object stream (same T as video frames)
            self.obj_tcn = DilatedTCN(
                d_model,
                layers=tcn_layers,
                kernel_size=k,
                dilations=dilations,
                dropout=dropout,
            )
            # Gate object stream into fused 2D+3D stream (for losses)
            self.obj_gate = nn.Linear(2 * d_model, d_model)
            self.ln_fuse_all = nn.LayerNorm(d_model)
        else:
            self.obj_to_gamma_beta = None
            self.obj_ln = None
            self.obj_align_proj = None
            self.obj_align_ln = None
            self.obj_tcn = None
            self.obj_gate = None
            self.ln_fuse_all = nn.Identity()

        # -------- Project 2D / 3D into shared d_model space --------
        self.proj2d = nn.Linear(feature_dim_2d, d_model)
        self.proj3d = nn.Linear(feature_dim_3d, d_model)
        self.ln2d = nn.LayerNorm(d_model)
        self.ln3d = nn.LayerNorm(d_model)

        # -------- Dual temporal encoders: 2D and 3D streams --------
        self.encoder2d = DilatedTCN(
            d_model,
            layers=tcn_layers,
            kernel_size=k,
            dilations=dilations,
            dropout=dropout,
        )
        self.encoder3d = DilatedTCN(
            d_model,
            layers=tcn_layers,
            kernel_size=k,
            dilations=dilations,
            dropout=dropout,
        )

        # -------- GMU-style gated fusion between 2D and 3D (for fused enc_states) --------
        self.h2d_t = nn.Linear(d_model, d_model)
        self.h3d_t = nn.Linear(d_model, d_model)
        self.gmu_z = nn.Linear(2 * d_model, d_model)
        self.ln_fuse = nn.LayerNorm(d_model)

        # -------- Alignment projection (encoder -> CLIP text space) --------
        if align_dim is None:
            align_dim = d_model
        self.align_dim = align_dim
        self.align_proj = nn.Linear(d_model, align_dim)

        # -------- Transformer decoder (takes 2D & 3D memories) --------
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=dec_layers,
            dim_feedforward=ff,
            dropout=dropout,
            pad_id=pad_id,
        )

    # ------------------------------------------------------------------
    #  Object + 2D fusion (FiLM-style, on raw 2D features)
    # ------------------------------------------------------------------
    def _fuse_obj_with_2d(
        self,
        feats2d: torch.Tensor,          # [B, T, D2]
        obj_feats: torch.Tensor | None  # [B, T, D_obj] or None
    ) -> torch.Tensor:
        """
        FiLM-style modulation:
            gamma, beta = MLP(obj)
            feats2d' = gamma * feats2d + beta
        """
        if not self.use_obj or obj_feats is None:
            return feats2d

        gamma_beta = self.obj_to_gamma_beta(obj_feats)  # [B, T, 2*D2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        fused = gamma * feats2d + beta
        return self.obj_ln(fused)

    # ------------------------------------------------------------------
    #  2D / 3D fusion (GMU-style, on top of dual TCN outputs)
    # ------------------------------------------------------------------
    def _fuse_2d_3d(
        self,
        h2d: torch.Tensor,  # [B, T, d_model]
        h3d: torch.Tensor,  # [B, T, d_model]
    ) -> torch.Tensor:
        """
        GMU-style fusion:
            h2d_t = tanh(W2 * h2d)
            h3d_t = tanh(W3 * h3d)
            z    = sigmoid(Wz * [h2d; h3d])
            h    = z ⊙ h2d_t + (1 - z) ⊙ h3d_t
        """
        h2d_t = torch.tanh(self.h2d_t(h2d))
        h3d_t = torch.tanh(self.h3d_t(h3d))

        concat = torch.cat([h2d, h3d], dim=-1)  # [B, T, 2*d_model]
        z = torch.sigmoid(self.gmu_z(concat))   # [B, T, d_model]

        fused = z * h2d_t + (1.0 - z) * h3d_t
        return self.ln_fuse(fused)

    # ------------------------------------------------------------------
    #  Internal: encode 2D / 3D / object streams (before fusion)
    # ------------------------------------------------------------------
    def _encode_streams(
        self,
        feats2d: torch.Tensor,                      # [B, T, D2]
        feats3d: torch.Tensor,                      # [B, T, D3]
        feat_pad_mask: torch.Tensor | None = None,  # [B, T] bool
        obj_feats: torch.Tensor | None = None,      # [B, T, D_obj] or None
    ):
        """
        Compute per-stream encodings:
          - FiLM-modulated 2D -> proj -> 2D TCN
          - 3D -> proj -> 3D TCN (can be disabled)
          - optional object stream -> proj -> object TCN
        Returns:
          h2d: [B, T, d_model]
          h3d: [B, T, d_model]
          obj_aligned: [B, T, d_model] or None
          obj_temporal: [B, T, d_model] or None
        """
        # Obj → 2D (FiLM)
        feats2d_mod = self._fuse_obj_with_2d(feats2d, obj_feats)  # [B, T, D2]

        # Project to d_model
        h2d = self.ln2d(self.proj2d(feats2d_mod))  # [B, T, d_model]
        h3d = self.ln3d(self.proj3d(feats3d))      # [B, T, d_model]

        if h2d.size(1) != h3d.size(1):
            raise ValueError(
                f"T mismatch between 2D and 3D feats: {h2d.size(1)} vs {h3d.size(1)}"
            )

        # Temporal modeling for each stream
        h2d = self.encoder2d(h2d)  # [B, T, d_model]
        h3d = self.encoder3d(h3d)  # [B, T, d_model]

        # If 3D stream is disabled (2D-only ablation), zero it out
        if not self.use_3d_stream:
            h3d = torch.zeros_like(h2d)

        obj_aligned = None
        obj_temporal = None
        if self.use_obj and obj_feats is not None:
            # Object-aligned representation (for auxiliary loss)
            obj_aligned = self.obj_align_ln(self.obj_align_proj(obj_feats))  # [B, T, d_model]
            # Temporal modeling of object stream
            obj_temporal = self.obj_tcn(obj_aligned)                         # [B, T, d_model]

        return h2d, h3d, obj_aligned, obj_temporal

    # ------------------------------------------------------------------
    #  Encode video features (for losses / CLIP / SCST)
    # ------------------------------------------------------------------
    def encode(
        self,
        feats2d: torch.Tensor,                      # [B, T, D2]
        feats3d: torch.Tensor,                      # [B, T, D3]
        feat_pad_mask: torch.Tensor | None = None,  # [B, T] bool
        obj_feats: torch.Tensor | None = None,      # [B, T, D_obj] or None
        return_obj_align: bool = False,             # whether to return obj-aligned states
    ):
        """
        Encode video into a fused representation (for CLIP / object losses / SCST):
          - dual TCN streams (2D / 3D)
          - GMU fusion
          - optional object temporal stream + gated fusion
        Returns:
          enc_states: [B, T, d_model] fused representation
          feat_pad_mask: [B, T] bool
          obj_aligned: [B, T, d_model] or None
        """
        h2d, h3d, obj_aligned, obj_temporal = self._encode_streams(
            feats2d, feats3d, feat_pad_mask, obj_feats
        )

        # GMU fusion of 2D / 3D
        h_fused = self._fuse_2d_3d(h2d, h3d)  # [B, T, d_model]

        # Optional object gating into fused representation
        if self.use_obj and obj_temporal is not None:
            gate = torch.sigmoid(
                self.obj_gate(torch.cat([h_fused, obj_temporal], dim=-1))
            )  # [B, T, d_model]
            h_fused = self.ln_fuse_all(h_fused + gate * obj_temporal)

            if not return_obj_align:
                obj_aligned = None
        elif not return_obj_align:
            obj_aligned = None

        return h_fused, feat_pad_mask, obj_aligned

    # ------------------------------------------------------------------
    #  Forward (training / teacher forcing)
    # ------------------------------------------------------------------
    def forward(
        self,
        feats2d: torch.Tensor,                      # [B, T, D2]
        feats3d: torch.Tensor,                      # [B, T, D3]
        tgt_ids: torch.Tensor,                      # [B, L-1]
        feat_pad_mask: torch.Tensor | None = None,  # [B, T] bool
        tgt_pad_mask: torch.Tensor | None = None,   # [B, L-1] bool
        obj_feats: torch.Tensor | None = None,      # [B, T, D_obj] or None
        return_enc_states: bool = False,            # whether to return encoder states
    ):
        """
        Forward pass.
        - return_enc_states=False: return logits
        - return_enc_states=True: return (logits, enc_states, obj_aligned, mem_pad)
          where enc_states is the fused 2D+3D(+obj) representation used for losses.
        """
        # 1) Encode individual streams
        h2d, h3d, obj_aligned, obj_temporal = self._encode_streams(
            feats2d, feats3d, feat_pad_mask, obj_feats
        )

        # 2) Build fused enc_states for CLIP / object losses
        h_fused = self._fuse_2d_3d(h2d, h3d)  # [B, T, d_model]
        if self.use_obj and obj_temporal is not None:
            gate = torch.sigmoid(
                self.obj_gate(torch.cat([h_fused, obj_temporal], dim=-1))
            )
            h_fused = self.ln_fuse_all(h_fused + gate * obj_temporal)

        # If caller does not need obj_aligned, drop it here
        if not return_enc_states:
            obj_aligned_ret = None
        else:
            obj_aligned_ret = obj_aligned

        # 3) Decode with cross-attention to 2D / 3D memories
        logits = self.decoder(
            tgt_ids,
            mem_2d=h2d,
            mem_3d=h3d,
            tgt_key_padding_mask=tgt_pad_mask,
            mem_2d_key_padding_mask=feat_pad_mask,
            mem_3d_key_padding_mask=feat_pad_mask,
        )

        if return_enc_states:
            # mem_pad is the same frame-level mask
            return logits, h_fused, obj_aligned_ret, feat_pad_mask
        return logits
