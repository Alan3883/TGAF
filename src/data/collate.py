import torch


def pad_sequence(seqs, pad_value=0):
    """
    Pad sequences to the same length.
    
    Args:
        seqs: List of tensors with shape [L, ...]
        pad_value: Value to use for padding
        
    Returns:
        padded: Tensor of shape [B, max_len, ...]
        lengths: Tensor of shape [B] with actual lengths
    """
    lens = [s.size(0) for s in seqs]
    maxlen = max(lens) if lens else 0
    if maxlen == 0:
        return torch.empty(0), torch.tensor(lens)
    out = seqs[0].new_full((len(seqs), maxlen, *seqs[0].shape[1:]), pad_value)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s
    return out, torch.tensor(lens, dtype=torch.long)


def collate_fn(batch, pad_id):
    """
    Collate function for video captioning batches.
    
    Args:
        batch: List of dicts with keys
            ["feats2d", "feats3d", "obj_feats", "feat_mask", "tgt", "raw_caption"]
        pad_id: Padding token ID
        
    Returns:
        feats2d:   [B, T, D2] padded 2D features
        feats3d:   [B, T, D3] padded 3D features
        obj_feats: [B, T, D_obj] padded object features (or None if absent)
        feat_mask: [B, T] boolean mask (True for padding)
        y_in:      [B, L-1] input tokens for teacher forcing
        y_out:     [B, L-1] target tokens for teacher forcing
        tgt_mask:  [B, L-1] boolean mask for target (True for padding)
        raw_caps:  list of length B, each is a raw caption string
    """
    feats2d_list = [b["feats2d"] for b in batch]
    feats3d_list = [b["feats3d"] for b in batch]
    obj_list = [b.get("obj_feats") for b in batch]
    feat_mask_list = [b["feat_mask"] for b in batch]
    tgts_list = [b["tgt"] for b in batch]
    raw_caps = [b["raw_caption"] for b in batch]
    
    # Pad features: [B, T, D]
    feats2d_pad, _ = pad_sequence(feats2d_list, pad_value=0.0)
    feats3d_pad, _ = pad_sequence(feats3d_list, pad_value=0.0)
    if any(o is not None for o in obj_list):
        first_dim = next((o.size(-1) for o in obj_list if o is not None), None)
        filled = []
        for feats, obj in zip(feats2d_list, obj_list):
            if obj is None:
                obj = torch.zeros(feats.size(0), first_dim, dtype=feats.dtype)
            filled.append(obj)
        obj_pad, _ = pad_sequence(filled, pad_value=0.0)
    else:
        obj_pad = None

    # Pad masks to [B, T]
    feat_mask_pad, _ = pad_sequence(feat_mask_list, pad_value=True)
    
    # Pad targets: [B, L]
    tgts_pad, _ = pad_sequence(tgts_list, pad_value=pad_id)
    
    y_in = tgts_pad[:, :-1]  # [B, L-1]
    y_out = tgts_pad[:, 1:]  # [B, L-1]
    tgt_mask = (y_out == pad_id)  # [B, L-1]
    
    return feats2d_pad, feats3d_pad, obj_pad, feat_mask_pad, y_in, y_out, tgt_mask, raw_caps
