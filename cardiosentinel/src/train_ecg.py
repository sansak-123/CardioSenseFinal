"""
Trains the ECG risk classifier (Experiment 2) — separate from Experiment 1,
not fused into it. Run from inside cardiosentinel/:

    python -m src.train_ecg --config configs/ecg_smoke.yaml   # CPU correctness run
    python -m src.train_ecg --config configs/ecg.yaml         # real GPU training run

Uses PTB-XL's own shipped `strat_fold` stratified split (fold 9 = val, fold
10 = test by default), not a custom one.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from .data.ptbxl_dataset import (
    load_ptbxl_metadata, assign_superclass_labels, load_ptbxl_plus_features,
    ensure_demographics, official_fold_split, fit_normalization, apply_normalization,
    PTBXLDataset, DEMOGRAPHIC_COLS,
)
from .models.ecg_model import ECGRiskClassifier, PTBXL_SUPERCLASSES
from .utils.metrics import per_class_auroc
from .utils.seed import set_seed
from .paths import CARDIOSENTINEL_ROOT, PTBXL_SIGNAL_DIR


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available on this machine — falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def build_data(config: dict):
    database, scp_statements = load_ptbxl_metadata()
    labels = assign_superclass_labels(database, scp_statements)

    features = load_ptbxl_plus_features()
    features = ensure_demographics(features, database)

    common_ids = database.index.intersection(features.index)
    print(f"[ecg] {len(common_ids)}/{len(database)} records have PTB-XL+ features")
    database = database.loc[common_ids]
    labels = labels.loc[common_ids]
    features = features.loc[common_ids]

    if config.get("max_records"):
        common_ids = common_ids[:config["max_records"]]
        database, labels, features = database.loc[common_ids], labels.loc[common_ids], features.loc[common_ids]

    train_ids, val_ids, test_ids = official_fold_split(database, config["val_fold"], config["test_fold"])

    feature_cols = [c for c in features.columns if c not in DEMOGRAPHIC_COLS]
    stats = fit_normalization(features.loc[train_ids], feature_cols + DEMOGRAPHIC_COLS)
    features_norm = apply_normalization(features, stats)

    n_features = len(feature_cols)
    n_demo = len(DEMOGRAPHIC_COLS)

    make_ds = lambda ids: PTBXLDataset(ids, database, labels, features_norm, PTBXL_SIGNAL_DIR)
    return make_ds(train_ids), make_ds(val_ids), make_ds(test_ids), n_features, n_demo


def train_one_epoch(model, loader, optimizer, device, log_every: int = 20) -> float:
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, n_batches = 0.0, 0
    n_total_batches = len(loader)
    t0 = time.time()
    for batch in loader:
        signal = batch["signal"].to(device)
        features = batch["features"].to(device)
        demographics = batch["demographics"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(signal, features, demographics)
        loss = loss_fn(logits, label)
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
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_true, all_score = [], []
    for batch in loader:
        signal = batch["signal"].to(device)
        features = batch["features"].to(device)
        demographics = batch["demographics"].to(device)
        logits = model(signal, features, demographics)
        all_true.append(batch["label"].numpy())
        all_score.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(all_true)
    y_score = np.concatenate(all_score)
    return per_class_auroc(y_true, y_score, PTBXL_SUPERCLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    device = resolve_device(config["device"])

    train_ds, val_ds, test_ds, n_features, n_demo = build_data(config)
    print(f"records: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}, "
          f"n_ecg_features={n_features}, n_demo_features={n_demo}")

    num_workers = config.get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, num_workers=num_workers)

    model = ECGRiskClassifier(
        n_leads=12, n_ecg_features=n_features, n_demo_features=n_demo,
        signal_channels=config["signal_channels"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    for epoch in range(config["epochs"]):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_auroc = evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{config['epochs']}  train_loss={train_loss:.4f}  "
              f"val_auroc={val_auroc}  ({time.time() - t0:.1f}s)")

    test_auroc = evaluate(model, test_loader, device)
    print(f"\nTEST per-class AUROC: {test_auroc}")

    ckpt_dir = CARDIOSENTINEL_ROOT / config["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "ecg_risk_classifier.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
