"""
Finds and loads whichever trained checkpoints actually exist, reconstructing
each model with the architecture hyperparameters from the matching config
(checkpoint_dir name == config file stem, e.g. checkpoints/ecg/ <-> configs/ecg.yaml)
rather than hard-coding shapes — so this keeps working whether the checkpoint
came from a smoke run or a real one, without needing to modify the training
scripts to persist a sidecar file.

Both loaders return None (rather than raising) when no checkpoint exists yet,
so the API can start up and report "not trained yet" instead of crashing —
useful given training is still in progress when this is being built.
"""
import functools
from pathlib import Path

import torch
import yaml

from src.paths import CARDIOSENTINEL_ROOT
from src.data.stream1_ehr import VITAL_PARAMS
from src.data.ptbxl_dataset import DEMOGRAPHIC_COLS
from src.models.text_encoder import build_clinicalbert, build_biobert
from src.models.trimodal_model import TrimodalDeteriorationModel
from src.models.ecg_model import ECGRiskClassifier

# Confirmed by direct inspection of the actual downloaded ECGdeli features file
# during training (train_ecg.py logged "n_ecg_features=531, n_demo_features=4").
# Not persisted anywhere else, so hard-coded here — must be kept in sync if the
# feature source ever changes.
N_ECG_FEATURES = 531

# Population stats from ptbxl_database.csv (measured directly via pandas
# .describe() during data inspection), used to normalize demographic inputs
# for live requests the same way training-time z-scoring would have.
DEMOGRAPHIC_STATS = {
    "age":    {"mean": 59.836, "std": 16.953},
    "sex":    {"mean": 0.479,  "std": 0.500},
    "height": {"mean": 166.708, "std": 10.865},
    "weight": {"mean": 70.998,  "std": 15.875},
}


def _latest_checkpoint(ckpt_dir: Path):
    if not ckpt_dir.exists():
        return None
    candidates = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_config(name: str) -> dict:
    with open(CARDIOSENTINEL_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


@functools.lru_cache(maxsize=1)
def load_trimodal_model():
    """Returns (model, checkpoint_path, is_smoke) or (None, None, None)."""
    for name in ("trimodal", "trimodal_smoke"):
        ckpt_path = _latest_checkpoint(CARDIOSENTINEL_ROOT / "checkpoints" / name)
        if ckpt_path is None:
            continue
        config = _load_config(name)
        notes_encoder = build_clinicalbert(
            freeze=config["freeze_text_encoders"], max_length=config["notes_max_length"])
        symptoms_encoder = build_biobert(
            freeze=config["freeze_text_encoders"], max_length=config["symptoms_max_length"])
        model = TrimodalDeteriorationModel(
            n_ehr_features=len(VITAL_PARAMS), stream2_encoder=notes_encoder,
            stream3_encoder=symptoms_encoder, d_model=config["d_model"],
            ehr_d_model=config["ehr_d_model"], ehr_layers=config["ehr_layers"],
            ehr_heads=config["ehr_heads"], fusion_type=config["fusion_type"],
        )
        state = torch.load(ckpt_path, map_location="cpu")
        # Resumable runs save {"epoch", "model_state_dict", "optimizer_state_dict"};
        # older/smoke checkpoints predate that and are a bare state_dict.
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        return model, ckpt_path, (name == "trimodal_smoke")
    return None, None, None


@functools.lru_cache(maxsize=1)
def load_ecg_model():
    """Returns (model, checkpoint_path, is_smoke) or (None, None, None)."""
    for name in ("ecg", "ecg_smoke"):
        ckpt_path = _latest_checkpoint(CARDIOSENTINEL_ROOT / "checkpoints" / name)
        if ckpt_path is None:
            continue
        config = _load_config(name)
        model = ECGRiskClassifier(
            n_leads=12, n_ecg_features=N_ECG_FEATURES, n_demo_features=len(DEMOGRAPHIC_COLS),
            signal_channels=config["signal_channels"],
        )
        state = torch.load(ckpt_path, map_location="cpu")
        # Resumable runs save {"epoch", "model_state_dict", "optimizer_state_dict"};
        # older/smoke checkpoints predate that and are a bare state_dict.
        model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
        model.eval()
        return model, ckpt_path, (name == "ecg_smoke")
    return None, None, None
