"""
Cross-modal fusion for the trimodal deterioration model.

Two stages:
1. Cross-modal attention: the three stream embeddings are stacked into a
   length-3 sequence and passed through a small self-attention block, so each
   stream's representation is refined by attending over the other two (and
   itself) before being combined.
2. Adaptive Modality-Trust (AMT) gate: a small learned network that looks at
   the (refined) embeddings plus an explicit per-sample "is this modality
   present" flag and outputs a softmax weight per stream. Presence flags (not
   just the raw embedding, which is zeroed when missing) are what let the gate
   degrade gracefully instead of being confused by a literal zero vector.
"""
import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, stream_embeds: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        # stream_embeds: [B, 3, D] -> [B, 3, D], each token refined by attending
        # over all three (self-attention over the 3-modality sequence).
        # A missing stream is excluded as a KEY/VALUE via key_padding_mask, not
        # just zero-valued — otherwise the other two streams' refined
        # embeddings would still be pulled toward a fabricated zero vector
        # during attention, undermining the missing-modality ablation even
        # though the AMT gate itself would assign that stream zero weight.
        key_padding_mask = presence_mask == 0  # [B, 3], True = ignore as key/value
        return self.encoder(stream_embeds, src_key_padding_mask=key_padding_mask)


class AMTGate(nn.Module):
    """Per-sample softmax weights over the 3 streams, conditioned on the
    embeddings themselves and an explicit presence mask."""

    def __init__(self, d_model: int, n_streams: int = 3, hidden: int = 64):
        super().__init__()
        self.n_streams = n_streams
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * n_streams + n_streams, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_streams),
        )

    def forward(self, stream_embeds: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        # stream_embeds: [B, S, D], presence_mask: [B, S] (1 = present, 0 = missing)
        b, s, d = stream_embeds.shape
        flat = stream_embeds.reshape(b, s * d)
        gate_input = torch.cat([flat, presence_mask], dim=-1)
        logits = self.gate_mlp(gate_input)  # [B, S]
        # Never let a genuinely-missing modality receive positive weight.
        logits = logits.masked_fill(presence_mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # [B, S]
        return weights


class AMTFusion(nn.Module):
    def __init__(self, d_model: int, n_streams: int = 3, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.cross_attn = CrossModalAttention(d_model, n_heads=n_heads, n_layers=n_layers)
        self.gate = AMTGate(d_model, n_streams=n_streams)
        self.output_dim = d_model

    def forward(self, stream_embeds: torch.Tensor, presence_mask: torch.Tensor) -> torch.Tensor:
        refined = self.cross_attn(stream_embeds, presence_mask)  # [B, S, D]
        weights = self.gate(refined, presence_mask)             # [B, S]
        fused = (refined * weights.unsqueeze(-1)).sum(dim=1)     # [B, D]
        return fused, weights


class FixedWeightFusion(nn.Module):
    """Ablation baseline: uniform (non-adaptive) averaging over present streams,
    still preceded by the same cross-modal attention block so the comparison
    isolates the AMT gate's contribution rather than the attention stage."""

    def __init__(self, d_model: int, n_streams: int = 3, n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.cross_attn = CrossModalAttention(d_model, n_heads=n_heads, n_layers=n_layers)
        self.n_streams = n_streams
        self.output_dim = d_model

    def forward(self, stream_embeds: torch.Tensor, presence_mask: torch.Tensor):
        refined = self.cross_attn(stream_embeds, presence_mask)
        weights = presence_mask / presence_mask.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        fused = (refined * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights
