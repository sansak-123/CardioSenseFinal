"""
Full trimodal deterioration model: three stream encoders -> per-stream
projection to a shared dimension -> cross-modal attention + AMT (or fixed)
fusion -> three sigmoid heads (P_24h, P_48h, P_72h) sharing the fused vector.

`presence_mask` ([B, 3], 1=present/0=missing) is the single knob used for the
missing-modality ablation: setting a column to 0 zeroes that stream's embedding
before fusion AND tells the gate it's absent, regardless of what raw input was
passed in — so ablating a stream is one flag, not three code paths.
"""
import torch
import torch.nn as nn

from .ft_transformer import Stream1Encoder
from .text_encoder import PretrainedTextEncoder
from .fusion import AMTFusion, FixedWeightFusion


class TrimodalDeteriorationModel(nn.Module):
    def __init__(self, n_ehr_features: int, stream2_encoder: PretrainedTextEncoder,
                 stream3_encoder: PretrainedTextEncoder, d_model: int = 128,
                 ehr_d_model: int = 128, ehr_layers: int = 3, ehr_heads: int = 8,
                 fusion_type: str = "amt", horizons=("24h", "48h", "72h")):
        super().__init__()
        assert fusion_type in ("amt", "fixed")
        self.horizons = list(horizons)

        self.stream1_encoder = Stream1Encoder(
            n_ehr_features, d_model=ehr_d_model, n_layers=ehr_layers, n_heads=ehr_heads,
        )
        self.stream2_encoder = stream2_encoder
        self.stream3_encoder = stream3_encoder

        self.proj1 = nn.Linear(self.stream1_encoder.output_dim, d_model)
        self.proj2 = nn.Linear(self.stream2_encoder.output_dim, d_model)
        self.proj3 = nn.Linear(self.stream3_encoder.output_dim, d_model)

        fusion_cls = AMTFusion if fusion_type == "amt" else FixedWeightFusion
        self.fusion = fusion_cls(d_model, n_streams=3)

        self.heads = nn.ModuleDict({
            h: nn.Linear(self.fusion.output_dim, 1) for h in self.horizons
        })

    def forward(self, ehr_values, ehr_missing_mask, ehr_time_valid_mask,
                notes_input_ids, notes_attention_mask,
                symptoms_input_ids, symptoms_attention_mask,
                presence_mask: torch.Tensor):
        """
        presence_mask: [B, 3] float tensor, columns ordered (stream1, stream2, stream3).
        Returns: dict of horizon -> logits [B], plus 'gate_weights' [B, 3].
        """
        e1 = self.proj1(self.stream1_encoder(ehr_values, ehr_missing_mask, ehr_time_valid_mask))
        e2 = self.proj2(self.stream2_encoder(notes_input_ids, notes_attention_mask))
        e3 = self.proj3(self.stream3_encoder(symptoms_input_ids, symptoms_attention_mask))

        stream_embeds = torch.stack([e1, e2, e3], dim=1)  # [B, 3, D]
        stream_embeds = stream_embeds * presence_mask.unsqueeze(-1)

        fused, gate_weights = self.fusion(stream_embeds, presence_mask)

        logits = {h: self.heads[h](fused).squeeze(-1) for h in self.horizons}
        logits["gate_weights"] = gate_weights
        return logits
