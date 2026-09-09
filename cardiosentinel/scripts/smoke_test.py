"""
CPU-runnable correctness smoke test — proves both architectures are wired up
correctly (shapes, gradient flow, missing-modality ablation, end-to-end
trainability) without needing real data or a GPU. This is NOT a training run:
it uses random tensors and a tiny randomly-initialized text encoder (no
HuggingFace download) so it finishes in seconds on a laptop CPU. Real training
targets configs/*.yaml (non-smoke) on a GPU — see README.md.
"""
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.trimodal_model import TrimodalDeteriorationModel
from src.models.ecg_model import ECGRiskClassifier, PTBXL_SUPERCLASSES
from src.utils.seed import set_seed


class TinyTextEncoderForTesting(nn.Module):
    """Stand-in for PretrainedTextEncoder with the same forward interface
    (input_ids, attention_mask) -> [CLS] pooled embedding, but randomly
    initialized and tiny, so the smoke test needs no network access."""

    def __init__(self, vocab_size: int = 500, hidden_size: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=2, dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_dim = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        key_padding_mask = attention_mask == 0
        out = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return out[:, 0, :]


def check_grad_flow(model: nn.Module, name: str) -> None:
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise RuntimeError(f"[{name}] {len(missing)} params got no gradient, e.g. {missing[:5]}")
    print(f"  [{name}] gradient flow OK ({sum(1 for _ in model.parameters())} params checked)")


def smoke_test_trimodal() -> None:
    print("\n=== Trimodal deterioration model ===")
    set_seed(0)
    batch, hours, n_features = 4, 6, 10

    model = TrimodalDeteriorationModel(
        n_ehr_features=n_features,
        stream2_encoder=TinyTextEncoderForTesting(),
        stream3_encoder=TinyTextEncoderForTesting(),
        d_model=16, ehr_d_model=16, ehr_layers=2, ehr_heads=2,
        fusion_type="amt",
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    def random_batch(presence_mask):
        return dict(
            ehr_values=torch.randn(batch, hours, n_features),
            ehr_missing_mask=(torch.rand(batch, hours, n_features) < 0.2).float(),
            ehr_time_valid_mask=torch.ones(batch, hours),
            notes_input_ids=torch.randint(0, 500, (batch, 20)),
            notes_attention_mask=torch.ones(batch, 20),
            symptoms_input_ids=torch.randint(0, 500, (batch, 15)),
            symptoms_attention_mask=torch.ones(batch, 15),
            presence_mask=presence_mask,
        )

    # 1. Full forward + backward with all streams present.
    full_presence = torch.ones(batch, 3)
    out = model(**random_batch(full_presence))
    targets = {h: torch.randint(0, 2, (batch,)).float() for h in model.horizons}
    loss = sum(loss_fn(out[h], targets[h]) for h in model.horizons)
    loss.backward()
    check_grad_flow(model, "trimodal/all-present")
    assert not torch.isnan(loss), "loss is NaN with all streams present"
    print(f"  forward+backward OK, loss={loss.item():.4f}, gate_weights[0]={out['gate_weights'][0].tolist()}")

    opt.step()
    opt.zero_grad()

    # 2. Missing-modality ablation: zero out stream 2 (notes) via presence_mask.
    for missing_idx, stream_name in enumerate(["stream1", "stream2", "stream3"]):
        presence = torch.ones(batch, 3)
        presence[:, missing_idx] = 0.0
        out = model(**random_batch(presence))
        loss = sum(loss_fn(out[h], targets[h]) for h in model.horizons)
        assert not torch.isnan(loss), f"loss is NaN with {stream_name} ablated"
        gw = out["gate_weights"]
        assert torch.allclose(gw[:, missing_idx], torch.zeros(batch), atol=1e-6), \
            f"AMT gate gave nonzero weight to ablated {stream_name}"
        print(f"  ablation ({stream_name} zeroed) OK, loss={loss.item():.4f}, "
              f"gate_weights[0]={gw[0].tolist()}")

    # 3. Fixed-weight fusion baseline builds and runs too.
    fixed_model = TrimodalDeteriorationModel(
        n_ehr_features=n_features,
        stream2_encoder=TinyTextEncoderForTesting(),
        stream3_encoder=TinyTextEncoderForTesting(),
        d_model=16, ehr_d_model=16, ehr_layers=2, ehr_heads=2,
        fusion_type="fixed",
    )
    out = fixed_model(**random_batch(full_presence))
    assert torch.allclose(out["gate_weights"][0], torch.full((3,), 1 / 3), atol=1e-5)
    print("  fixed-weight fusion baseline OK (uniform weights confirmed)")


def smoke_test_ecg() -> None:
    print("\n=== ECG risk classifier ===")
    set_seed(0)
    batch, n_leads, n_samples = 4, 12, 1000
    n_ecg_features, n_demo_features = 20, 4

    model = ECGRiskClassifier(
        n_leads=n_leads, n_ecg_features=n_ecg_features, n_demo_features=n_demo_features,
        signal_channels=8,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    signal = torch.randn(batch, n_leads, n_samples)
    features = torch.randn(batch, n_ecg_features)
    demographics = torch.randn(batch, n_demo_features)
    targets = torch.randint(0, 2, (batch, len(PTBXL_SUPERCLASSES))).float()

    logits = model(signal, features, demographics)
    assert logits.shape == (batch, len(PTBXL_SUPERCLASSES)), logits.shape
    loss = loss_fn(logits, targets)
    loss.backward()
    check_grad_flow(model, "ecg")
    assert not torch.isnan(loss)
    print(f"  forward+backward OK, loss={loss.item():.4f}, logits.shape={tuple(logits.shape)}")
    opt.step()


if __name__ == "__main__":
    start = time.time()
    smoke_test_trimodal()
    smoke_test_ecg()
    print(f"\nSMOKE TEST PASSED in {time.time() - start:.1f}s")
