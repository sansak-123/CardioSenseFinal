"""
PyTorch Dataset for Experiment 1. One sample = one (stay_id, hour=t) row from
the synthetically-paired joint table. The EHR input exposes the full 48-hour
panel for that stay but with `ehr_time_valid_mask` zeroed out for hours > t —
this is what makes the "predict deterioration by t+h using only data up to t"
setup causally correct (a stay's hours after t never influence the pooled
embedding, since AttentionPool masks them to zero weight).

Text batching is intentionally NOT tokenized per-item in __getitem__ (slow,
one string at a time); `make_collate_fn` batches raw strings and calls the
tokenizer once per batch.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from .stream1_ehr import VITAL_PARAMS

HORIZON_NAMES = ["24h", "48h", "72h"]


def build_stay_arrays(wide_df, feature_cols=VITAL_PARAMS, max_hour: int = 48) -> dict:
    """wide_df: output of stream1_ehr.build_hourly_wide (one row per stay/hour,
    reindexed 0..max_hour-1). Returns {stay_id: (values[T,F] float32, mask[T,F] float32)}."""
    value_cols = feature_cols
    mask_cols = [f"{c}_missing" for c in feature_cols]
    arrays = {}
    for stay_id, group in wide_df.sort_values("hour").groupby("stay_id"):
        group = group.set_index("hour").reindex(range(max_hour))
        values = group[value_cols].fillna(0.0).to_numpy(dtype=np.float32)
        mask = group[mask_cols].fillna(1.0).to_numpy(dtype=np.float32)  # unindexed hour -> treat as missing
        arrays[stay_id] = (values, mask)
    return arrays


class TrimodalDeteriorationDataset(Dataset):
    def __init__(self, joint_df, stay_arrays: dict, max_hour: int = 48,
                 feature_cols=VITAL_PARAMS):
        self.joint_df = joint_df.reset_index(drop=True)
        self.stay_arrays = stay_arrays
        self.max_hour = max_hour
        self.n_features = len(feature_cols)

    def __len__(self) -> int:
        return len(self.joint_df)

    def __getitem__(self, idx: int) -> dict:
        row = self.joint_df.iloc[idx]
        values, mask = self.stay_arrays[row["stay_id"]]
        t = int(row["hour"])
        time_valid_mask = (np.arange(self.max_hour) <= t).astype(np.float32)

        labels = np.array([row[f"deteriorate_{h}"] for h in HORIZON_NAMES], dtype=np.float32)

        return dict(
            ehr_values=values,
            ehr_missing_mask=mask,
            ehr_time_valid_mask=time_valid_mask,
            note_text=row["paired_note_text"],
            symptom_text=row["paired_symptom_text"],
            labels=labels,
            stay_id=row["stay_id"],
            hour=t,
        )


def make_collate_fn(notes_encoder, symptoms_encoder):
    """notes_encoder/symptoms_encoder: PretrainedTextEncoder instances (used
    only for .tokenize() here — the actual forward pass happens in the model).

    Deliberately produces CPU tensors only, with no device transfer — with
    num_workers > 0 this function runs inside forked worker subprocesses, and
    CUDA cannot be re-initialized there (RuntimeError: "Cannot re-initialize
    CUDA in forked subprocess"). The training loop moves each batch to the
    target device itself, after it comes back from the DataLoader in the main
    process.
    """

    def collate(batch: list) -> dict:
        b = len(batch)
        ehr_values = torch.from_numpy(np.stack([x["ehr_values"] for x in batch]))
        ehr_missing_mask = torch.from_numpy(np.stack([x["ehr_missing_mask"] for x in batch]))
        ehr_time_valid_mask = torch.from_numpy(np.stack([x["ehr_time_valid_mask"] for x in batch]))
        labels = torch.from_numpy(np.stack([x["labels"] for x in batch]))

        notes_enc = notes_encoder.tokenize([x["note_text"] for x in batch], device=torch.device("cpu"))
        symptoms_enc = symptoms_encoder.tokenize([x["symptom_text"] for x in batch], device=torch.device("cpu"))

        return dict(
            ehr_values=ehr_values,
            ehr_missing_mask=ehr_missing_mask,
            ehr_time_valid_mask=ehr_time_valid_mask,
            notes_input_ids=notes_enc["input_ids"],
            notes_attention_mask=notes_enc["attention_mask"],
            symptoms_input_ids=symptoms_enc["input_ids"],
            symptoms_attention_mask=symptoms_enc["attention_mask"],
            presence_mask=torch.ones(b, 3),
            labels=labels,
        )

    return collate


def split_stay_ids(stay_ids: np.ndarray, val_frac: float = 0.15, test_frac: float = 0.15,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Splits at the ICU-stay level (never by row) so no timestep of a given
    stay leaks across splits."""
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(stay_ids)
    n = len(shuffled)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_ids = shuffled[:n_val]
    test_ids = shuffled[n_val:n_val + n_test]
    train_ids = shuffled[n_val + n_test:]
    return train_ids, val_ids, test_ids
