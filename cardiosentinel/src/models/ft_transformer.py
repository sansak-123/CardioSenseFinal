"""
Stream 1 (EHR/vitals) encoder: an FT-Transformer over per-hour numeric features,
with an explicit missingness-mask embedding per feature (ICU time series have
real gaps — a feature being absent is itself informative), followed by temporal
attention pooling across the hour sequence into a single vector.
"""
import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Turns a [*, n_features] numeric row (+ missingness mask) into a token
    sequence: one learned d_model embedding per feature, scaled by its value and
    additively perturbed by a per-feature "this was missing" embedding, plus a
    prepended [CLS] token."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        # Per-feature affine numeric tokenizer (Gorishniy et al., FT-Transformer).
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.missing_embed = nn.Parameter(torch.empty(n_features, d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.weight, std=1 / math.sqrt(d_model))
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.missing_embed, std=1 / math.sqrt(d_model))
        nn.init.normal_(self.cls_token, std=1 / math.sqrt(d_model))

    def forward(self, values: torch.Tensor, missing_mask: torch.Tensor) -> torch.Tensor:
        # values, missing_mask: [B, n_features]. Missing values arrive as 0 in
        # `values` (already imputed upstream) — the mask is what carries the
        # "was this actually observed" signal.
        values = values.masked_fill(missing_mask.bool(), 0.0)
        tokens = values.unsqueeze(-1) * self.weight + self.bias  # [B, F, D]
        tokens = tokens + missing_mask.unsqueeze(-1) * self.missing_embed
        cls = self.cls_token.expand(values.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)  # [B, 1+F, D]


class AttentionPool(nn.Module):
    """Single learned query attends over a padded time sequence -> one vector."""

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.query, std=1 / math.sqrt(d_model))
        self.scale = d_model ** -0.5

    def forward(self, seq: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # seq: [B, T, D], valid_mask: [B, T] (1 = real timestep, 0 = padding)
        scores = (seq @ self.query.transpose(-1, -2).expand(seq.size(0), -1, -1)).squeeze(-1)
        scores = scores * self.scale
        scores = scores.masked_fill(valid_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (seq * weights).sum(dim=1)  # [B, D]


class Stream1Encoder(nn.Module):
    """FT-Transformer per hour + temporal attention pooling across hours."""

    def __init__(self, n_features: int, d_model: int = 128, n_layers: int = 3,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.per_hour_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.temporal_pool = AttentionPool(d_model)
        self.output_dim = d_model

    def forward(self, values: torch.Tensor, missing_mask: torch.Tensor,
                time_valid_mask: torch.Tensor) -> torch.Tensor:
        """
        values, missing_mask: [B, T, F]
        time_valid_mask: [B, T] — 1 where hour t has real (possibly all-missing
            but present) data, 0 where the sequence was padded to a common length.
        Returns: [B, output_dim] pooled patient-stay embedding.
        """
        b, t, f = values.shape
        flat_values = values.reshape(b * t, f)
        flat_mask = missing_mask.reshape(b * t, f)
        tokens = self.tokenizer(flat_values, flat_mask)          # [B*T, 1+F, D]
        encoded = self.per_hour_transformer(tokens)               # [B*T, 1+F, D]
        per_hour_embed = encoded[:, 0, :].reshape(b, t, -1)        # [B, T, D] (CLS)
        return self.temporal_pool(per_hour_embed, time_valid_mask)
