"""
Trains the trimodal deterioration model (Experiment 1). Run from inside
cardiosentinel/ so `src` resolves as a package:

    python -m src.train_trimodal --config configs/trimodal_smoke.yaml   # CPU correctness run
    python -m src.train_trimodal --config configs/trimodal.yaml         # real GPU training run

Ablation flags:
    --ablate-stream {0,1,2}   zero stream1/2/3 at BOTH train and eval time
                              (component-contribution ablation — retrains a variant)
    --fusion {amt,fixed}      override config's fusion_type (AMT gate vs uniform baseline)
    --pairing-seed N          override config's pairing_seed (synthetic-pairing sensitivity)

Every run (unless itself an --ablate-stream run) finishes with a missing-
modality-at-inference sweep on the trained model: zero each stream in turn at
eval time only, no retraining, to check the AMT gate degrades gracefully
rather than collapsing — this is the "missing-modality robustness" design
goal from the spec, checked automatically rather than left as a one-off.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from .data.stream1_ehr import load_raw_long_format, build_hourly_wide, VITAL_PARAMS
from .data.labeling import add_deterioration_labels
from .data.stream2_notes import load_cardiac_notes
from .data.stream3_symptoms import load_stream3_raw
from .data.synthetic_pairing import build_joint_dataset
from .data.trimodal_dataset import (
    TrimodalDeteriorationDataset, build_stay_arrays, make_collate_fn, split_stay_ids,
)
from .models.text_encoder import build_clinicalbert, build_biobert
from .models.trimodal_model import TrimodalDeteriorationModel
from .utils.metrics import per_horizon_metrics
from .utils.seed import set_seed
from .paths import CARDIOSENTINEL_ROOT

STREAM_NAMES = ["stream1", "stream2", "stream3"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available on this machine — falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def build_data(config: dict):
    long_df, outcomes_df = load_raw_long_format()
    wide = build_hourly_wide(long_df, VITAL_PARAMS, max_hour=config["max_hour"])
    labeled = add_deterioration_labels(wide, outcomes_df)

    if config.get("max_stays"):
        keep = labeled["stay_id"].unique()[:config["max_stays"]]
        labeled = labeled[labeled["stay_id"].isin(keep)]
        wide = wide[wide["stay_id"].isin(keep)]

    notes = load_cardiac_notes()
    symptoms = load_stream3_raw()

    joint = build_joint_dataset(
        labeled, notes, symptoms,
        pairing_seed=config["pairing_seed"], temperature=config["pairing_temperature"],
    )
    stay_arrays = build_stay_arrays(wide, VITAL_PARAMS, config["max_hour"])
    dataset = TrimodalDeteriorationDataset(joint, stay_arrays, config["max_hour"], VITAL_PARAMS)

    train_ids, val_ids, test_ids = split_stay_ids(
        joint["stay_id"].unique(), config["val_frac"], config["test_frac"], seed=config["seed"],
    )
    idx_by_stay: dict = {}
    for i, sid in enumerate(joint["stay_id"].to_numpy()):
        idx_by_stay.setdefault(sid, []).append(i)

    def indices_for(ids):
        out = []
        for sid in ids:
            out.extend(idx_by_stay.get(sid, []))
        return out

    return dataset, indices_for(train_ids), indices_for(val_ids), indices_for(test_ids)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def run_batch(model, batch, device, presence_override_idx=None):
    batch = _move_batch(batch, device)
    presence = batch["presence_mask"].clone()
    if presence_override_idx is not None:
        presence[:, presence_override_idx] = 0.0
    out = model(
        batch["ehr_values"], batch["ehr_missing_mask"], batch["ehr_time_valid_mask"],
        batch["notes_input_ids"], batch["notes_attention_mask"],
        batch["symptoms_input_ids"], batch["symptoms_attention_mask"],
        presence,
    )
    out["labels"] = batch["labels"]
    return out


def train_one_epoch(model, loader, optimizer, device, ablate_stream_idx=None, log_every: int = 20) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, n_batches = 0.0, 0
    n_total_batches = len(loader)
    t0 = time.time()
    for batch in loader:
        optimizer.zero_grad()
        out = run_batch(model, batch, device, ablate_stream_idx)
        loss = sum(loss_fn(out[h], out["labels"][:, i]) for i, h in enumerate(model.horizons))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
        if n_batches % log_every == 0 or n_batches == n_total_batches:
            elapsed = time.time() - t0
            rate = n_batches / elapsed
            eta = (n_total_batches - n_batches) / rate if rate > 0 else float("inf")
            print(f"    batch {n_batches}/{n_total_batches}  loss={loss.item():.4f}  "
                  f"{rate:.2f} batch/s  elapsed={elapsed:.1f}s  eta_this_epoch={eta:.1f}s", flush=True)
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, zero_stream_idx=None) -> dict:
    model.eval()
    y_true = {h: [] for h in model.horizons}
    y_score = {h: [] for h in model.horizons}
    for batch in loader:
        out = run_batch(model, batch, device, zero_stream_idx)
        for i, h in enumerate(model.horizons):
            y_true[h].append(out["labels"][:, i].cpu().numpy())
            y_score[h].append(torch.sigmoid(out[h]).cpu().numpy())
    y_true = {h: np.concatenate(v) for h, v in y_true.items()}
    y_score = {h: np.concatenate(v) for h, v in y_score.items()}
    return per_horizon_metrics(y_true, y_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ablate-stream", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--fusion", choices=["amt", "fixed"], default=None)
    parser.add_argument("--pairing-seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.fusion:
        config["fusion_type"] = args.fusion
    if args.pairing_seed is not None:
        config["pairing_seed"] = args.pairing_seed

    set_seed(config["seed"])
    device = resolve_device(config["device"])

    dataset, train_idx, val_idx, test_idx = build_data(config)
    print(f"dataset: {len(dataset)} rows total | train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    notes_encoder = build_clinicalbert(freeze=config["freeze_text_encoders"], max_length=config["notes_max_length"]).to(device)
    symptoms_encoder = build_biobert(freeze=config["freeze_text_encoders"], max_length=config["symptoms_max_length"]).to(device)
    collate_fn = make_collate_fn(notes_encoder, symptoms_encoder)

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config["batch_size"],
                               shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config["batch_size"],
                             shuffle=False, collate_fn=collate_fn, num_workers=num_workers)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=config["batch_size"],
                              shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

    model = TrimodalDeteriorationModel(
        n_ehr_features=len(VITAL_PARAMS), stream2_encoder=notes_encoder, stream3_encoder=symptoms_encoder,
        d_model=config["d_model"], ehr_d_model=config["ehr_d_model"], ehr_layers=config["ehr_layers"],
        ehr_heads=config["ehr_heads"], fusion_type=config["fusion_type"],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    ablate_name = STREAM_NAMES[args.ablate_stream] if args.ablate_stream is not None else None
    run_name = f"trimodal_fusion-{config['fusion_type']}_pairing-{config['pairing_seed']}"
    if ablate_name:
        run_name += f"_ablate-{ablate_name}"
    print(f"=== Run: {run_name} (device={device}) ===")

    ckpt_dir = CARDIOSENTINEL_ROOT / config["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{run_name}.pt"

    # Saved every epoch (not just at the end) and resumable: a long GPU run on
    # a time-limited environment (e.g. free-tier Colab) can lose the session
    # partway through, and re-fine-tuning two BERT models from scratch would
    # waste hours of already-completed progress.
    start_epoch = 0
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from {ckpt_path} at epoch {start_epoch + 1}/{config['epochs']}")

    for epoch in range(start_epoch, config["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.ablate_stream)
        val_metrics = evaluate(model, val_loader, device, args.ablate_stream)
        print(f"epoch {epoch + 1}/{config['epochs']}  train_loss={train_loss:.4f}  "
              f"val={val_metrics}  ({time.time() - t0:.1f}s)")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, ckpt_path)
        print(f"  checkpoint saved ({ckpt_path})")

    test_metrics = evaluate(model, test_loader, device, args.ablate_stream)
    print(f"\nTEST metrics ({run_name}): {test_metrics}")

    if args.ablate_stream is None:
        print("\n-- missing-modality-at-inference sweep (same trained model, no retraining) --")
        for idx, name in enumerate(STREAM_NAMES):
            swept = evaluate(model, test_loader, device, idx)
            print(f"  zero {name} at inference: {swept}")


if __name__ == "__main__":
    main()
