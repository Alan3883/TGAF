import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDecoderLayer(nn.Module):
    """
    One Transformer decoder layer with:
      - self-attention on text tokens
      - two cross-attentions (2D / 3D video streams)
      - gate to fuse 2D / 3D contexts per token

    If return_gate=True, also output a scalar gate per token.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention over target tokens
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Cross-attention to 2D and 3D video memories
        self.cross_attn_2d = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn_3d = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Gate: fuse [ctx_2d, ctx_3d] -> fused_ctx
        self.gate_fc = nn.Linear(2 * d_model, d_model)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer norms + dropouts
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,                        # [B, L, C]
        mem_2d: torch.Tensor,                     # [B, T, C]
        mem_3d: torch.Tensor,                     # [B, T, C]
        tgt_mask: torch.Tensor | None = None,                # [L, L] bool (causal)
        tgt_key_padding_mask: torch.Tensor | None = None,    # [B, L] bool
        mem_2d_key_padding_mask: torch.Tensor | None = None, # [B, T] bool
        mem_3d_key_padding_mask: torch.Tensor | None = None, # [B, T] bool
        return_gate: bool = False,                            # if True, also return gate per token
    ):
        """
        Single decoder layer forward.

        If return_gate=True, returns (x, gate_per_token) where gate_per_token is [B, L].
        Otherwise returns x only.
        """

        # 1) Self-attention on target tokens (causal)
        x = tgt
        sa_out, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout1(sa_out)
        x = self.norm1(x)  # [B, L, C]

        # 2) Cross-attention to 2D memory
        ctx_2d, _ = self.cross_attn_2d(
            x,
            mem_2d,
            mem_2d,
            key_padding_mask=mem_2d_key_padding_mask,
        )  # [B, L, C]

        # 3) Cross-attention to 3D memory
        ctx_3d, _ = self.cross_attn_3d(
            x,
            mem_3d,
            mem_3d,
            key_padding_mask=mem_3d_key_padding_mask,
        )  # [B, L, C]

        # 4) GMU-style gate between 2D and 3D contexts
        ctx_cat = torch.cat([ctx_2d, ctx_3d], dim=-1)  # [B, L, 2C]
        gate = torch.sigmoid(self.gate_fc(ctx_cat))    # [B, L, C]
        fused_ctx = gate * ctx_2d + (1.0 - gate) * ctx_3d

        x = x + self.dropout2(fused_ctx)
        x = self.norm2(x)

        # 5) Position-wise FFN
        ffn = self.linear2(F.relu(self.linear1(x)))
        x = x + self.dropout3(ffn)
        x = self.norm3(x)  # [B, L, C]

        if return_gate:
            # average over channels -> one scalar gate per token
            # this is the weight for the 2D stream; 3D = 1 - gate
            gate_scalar = gate.mean(dim=-1)  # [B, L]
            return x, gate_scalar

        return x


class TransformerDecoder(nn.Module):
    """
    Transformer decoder for autoregressive caption generation.
    Uses:
      - text self-attention
      - two video memories (2D / 3D)
      - per-token gate to select 2D vs 3D context
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Hidden dimension
            nhead: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: FFN dimension
            dropout: Dropout probability
            pad_id: Padding token ID
        """
        super().__init__()

        # Token embedding (no positional encoding is changed here)
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        # Stacked gated decoder layers
        self.layers = nn.ModuleList(
            [
                GatedDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def _build_causal_mask(self, L: int, device: torch.device) -> torch.Tensor:
        """
        Create a [L, L] upper-triangular causal mask (True = masked).
        """
        return torch.triu(
            torch.ones(L, L, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def forward(
        self,
        tgt_ids: torch.Tensor,                        # [B, L]
        mem_2d: torch.Tensor,                         # [B, T, C]
        mem_3d: torch.Tensor,                         # [B, T, C]
        tgt_key_padding_mask: torch.Tensor | None = None,       # [B, L] bool
        mem_2d_key_padding_mask: torch.Tensor | None = None,    # [B, T] bool
        mem_3d_key_padding_mask: torch.Tensor | None = None,    # [B, T] bool
        return_gate_maps: bool = False,                          # if True, also return gate per token per layer
    ):
        """
        Forward pass for caption generation.

        Args:
            tgt_ids: [B, L] input tokens (teacher forcing)
            mem_2d: [B, T, C] encoder outputs from 2D stream
            mem_3d: [B, T, C] encoder outputs from 3D stream
            tgt_key_padding_mask: [B, L] True for padding tokens
            mem_2d_key_padding_mask: [B, T] True for padding frames in 2D
            mem_3d_key_padding_mask: [B, T] True for padding frames in 3D
            return_gate_maps: if True, also return list[num_layers] of [B, L] gates

        Returns:
            If return_gate_maps=False:
                logits: [B, L, vocab_size]
            If return_gate_maps=True:
                logits: [B, L, vocab_size]
                gate_maps: list of length num_layers, each [B, L]
        """
        # 1) Token embeddings
        tgt = self.embed(tgt_ids)  # [B, L, C]

        # 2) Causal mask for autoregressive decoding
        L = tgt.size(1)
        causal = self._build_causal_mask(L, tgt.device)  # [L, L]

        # 3) Stacked gated decoder layers
        x = tgt
        gate_maps = [] if return_gate_maps else None

        for layer in self.layers:
            if return_gate_maps:
                # Ask the layer to also return per-token gates
                x, gate_scalar = layer(
                    x,
                    mem_2d=mem_2d,
                    mem_3d=mem_3d,
                    tgt_mask=causal,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    mem_2d_key_padding_mask=mem_2d_key_padding_mask,
                    mem_3d_key_padding_mask=mem_3d_key_padding_mask,
                    return_gate=True,
                )
                gate_maps.append(gate_scalar)  # each [B, L]
            else:
                x = layer(
                    x,
                    mem_2d=mem_2d,
                    mem_3d=mem_3d,
                    tgt_mask=causal,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    mem_2d_key_padding_mask=mem_2d_key_padding_mask,
                    mem_3d_key_padding_mask=mem_3d_key_padding_mask,
                    return_gate=False,
                )

        x = self.ln(x)
        logits = self.out(x)  # [B, L, vocab_size]

        if return_gate_maps:
            return logits, gate_maps

        return logits
