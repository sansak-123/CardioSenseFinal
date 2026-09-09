"""
Experiment 2 data: PTB-XL raw waveforms (khyeh0719/ptb-xl-dataset, a mirror of
the standard PhysioNet PTB-XL release) + PTB-XL+ engineered ECG features
(antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset), joined on ecg_id.

Confirmed by inspection (scripts/inspect_schemas.py) against the actual downloaded copies:
  ptbxl_database.csv — ecg_id, patient_id, age, sex, height, weight, scp_codes,
                        strat_fold, filename_lr, filename_hr. This is also where
                        ALL demographics come from — height/weight are ~68%/57%
                        missing here, and PTB-XL+ has none at all (see below).
  scp_statements.csv — maps individual SCP codes -> diagnostic_class
                        (NORM/MI/STTC/CD/HYP)
  records100/**/*.hea + *.dat — WFDB waveform files at 100Hz (10s -> 1000 samples)
  PTB-XL+ — three separate ECG feature-extraction algorithms' outputs
    (features/12sl_features.csv, ecgdeli_features.csv, unig_features.csv), each
    keyed by ecg_id, pure ECG-morphology features — no demographics anywhere in it,
    despite the dataset's own description suggesting otherwise. ecgdeli_features.csv
    is used (open-source/reproducible, unlike the commercial 12SL black box).
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from torch.utils.data import Dataset

from ..paths import PTBXL_SIGNAL_DIR, PTBXL_FEATURES_DIR
from ..models.ecg_model import PTBXL_SUPERCLASSES

DEMOGRAPHIC_COLS = ["age", "sex", "height", "weight"]


def _find_file(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} anywhere under {root}.")
    return matches[0]


def load_ptbxl_metadata(root: Path = PTBXL_SIGNAL_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = _find_file(root, "ptbxl_database.csv")
    scp_path = _find_file(root, "scp_statements.csv")

    database = pd.read_csv(db_path, index_col="ecg_id")
    scp_statements = pd.read_csv(scp_path, index_col=0)

    print(f"[ptbxl] loaded {db_path.name}: shape={database.shape}")
    print(f"[ptbxl] columns: {list(database.columns)}")

    required = {"patient_id", "age", "sex", "height", "weight", "scp_codes", "strat_fold", "filename_lr"}
    missing = required - set(database.columns)
    if missing:
        raise ValueError(
            f"ptbxl_database.csv missing expected columns {missing}; "
            f"found {list(database.columns)}. Update src/data/ptbxl_dataset.py."
        )
    if "diagnostic_class" not in scp_statements.columns:
        raise ValueError(
            f"scp_statements.csv missing 'diagnostic_class'; found {list(scp_statements.columns)}."
        )

    database["scp_codes"] = database["scp_codes"].apply(ast.literal_eval)
    return database, scp_statements


def assign_superclass_labels(database: pd.DataFrame, scp_statements: pd.DataFrame) -> pd.DataFrame:
    """Adds one binary column per PTB-XL diagnostic superclass (multi-label —
    a record can map to more than one, e.g. NORM is mutually exclusive with
    the others but MI/STTC/CD/HYP can co-occur)."""
    diagnostic_scp = scp_statements[scp_statements["diagnostic_class"].notna()]
    code_to_class = diagnostic_scp["diagnostic_class"].to_dict()

    # Vectorized: explode (ecg_id, code) pairs once, map to superclass, then a
    # single crosstab — avoids ~2-3 scalar .loc writes per record (slow at
    # PTB-XL's ~21k-record scale).
    pairs = [(ecg_id, code) for ecg_id, codes in database["scp_codes"].items() for code in codes]
    codes_df = pd.DataFrame(pairs, columns=["ecg_id", "code"])
    codes_df["diagnostic_class"] = codes_df["code"].map(code_to_class)
    codes_df = codes_df[codes_df["diagnostic_class"].isin(PTBXL_SUPERCLASSES)]

    labels = pd.crosstab(codes_df["ecg_id"], codes_df["diagnostic_class"])
    labels = (labels > 0).reindex(index=database.index, columns=PTBXL_SUPERCLASSES, fill_value=False)
    labels = labels.astype(np.float32)

    has_label = labels.sum(axis=1) > 0
    print(f"[ptbxl] {has_label.sum()}/{len(labels)} records have >=1 recognized superclass label")
    return labels


# PTB-XL+ (antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset) actually ships
# THREE separate feature-extraction algorithms' outputs, no demographics at all:
#   features/12sl_features.csv     — commercial 12SL algorithm (black-box)
#   features/ecgdeli_features.csv  — open-source ECGdeli toolbox (chosen: reproducible,
#                                     peer-reviewed, not a proprietary black box)
#   features/unig_features.csv     — Uni-G algorithm
# (features/old/* are superseded duplicates of the same three — excluded.)
PTBXL_PLUS_FEATURE_FILE = "features/ecgdeli_features.csv"


def load_ptbxl_plus_features(root: Path = PTBXL_FEATURES_DIR) -> pd.DataFrame:
    matches = [p for p in root.rglob("ecgdeli_features.csv") if "old" not in p.parts]
    if not matches:
        raise FileNotFoundError(
            f"Expected {PTBXL_PLUS_FEATURE_FILE} under {root} but didn't find it. "
            "Run scripts/inspect_schemas.py to see the actual layout — this mirror's "
            "structure may differ; update PTBXL_PLUS_FEATURE_FILE in src/data/ptbxl_dataset.py."
        )
    path = matches[0]
    df = pd.read_csv(path)
    print(f"[ptbxl_plus] loaded {path.relative_to(root)}: shape={df.shape}")
    print(f"[ptbxl_plus] columns: {list(df.columns)[:30]}{'...' if df.shape[1] > 30 else ''}")

    id_candidates = [c for c in df.columns if c.lower() in ("ecg_id", "id", "record_id")]
    if not id_candidates:
        raise ValueError(
            f"Could not find an ecg_id-like column in PTB-XL+ features; found {list(df.columns)}. "
            "Update load_ptbxl_plus_features in src/data/ptbxl_dataset.py to name the right column."
        )
    df = df.rename(columns={id_candidates[0]: "ecg_id"}).set_index("ecg_id")
    return df


def ensure_demographics(features: pd.DataFrame, database: pd.DataFrame) -> pd.DataFrame:
    """PTB-XL+'s feature CSVs carry no demographics at all (confirmed by inspection —
    they're pure ECG-morphology features) — age/sex/height/weight all come from
    ptbxl_database.csv instead. height/weight are heavily missing there (~68%/57%
    NaN in the full dataset) — that's real data sparsity, not a bug; normalization
    (fit_normalization/apply_normalization) imputes with the train-set mean, so the
    model effectively sees "average" for most patients on those two features."""
    out = features.copy()
    lower_map = {c.lower(): c for c in out.columns}
    for col in DEMOGRAPHIC_COLS:
        if col not in lower_map:
            if col in database.columns:
                out[col] = database.reindex(out.index)[col]
                missing_rate = out[col].isna().mean()
                print(f"[ptbxl_plus] '{col}' not in PTB-XL+ features — filled from "
                      f"ptbxl_database.csv ({missing_rate:.0%} missing there)")
            else:
                raise ValueError(
                    f"Demographic column '{col}' not found in PTB-XL+ features or "
                    f"ptbxl_database.csv (database columns: {list(database.columns)}). "
                    "Update DEMOGRAPHIC_COLS / ensure_demographics in src/data/ptbxl_dataset.py."
                )
        elif lower_map[col] != col:
            out = out.rename(columns={lower_map[col]: col})
    return out


def resolve_signal_root(root: Path) -> Path:
    """Finds the directory that directly contains `records100/` — the Kaggle
    zip extracts into a nested subfolder (e.g.
    ptb-xl-a-large-...-1.0.1/records100/...), so `root` itself usually isn't
    it. Resolved ONCE per dataset, not per-sample: doing this search inside
    load_waveform (as a per-item fallback) was the actual cause of a ~40x
    slowdown — every __getitem__ call fell back to an uncached
    root.rglob(...) over 40,000+ files just to find one record."""
    if (root / "records100").is_dir():
        return root
    matches = [p for p in root.iterdir() if p.is_dir() and (p / "records100").is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a 'records100' directory under {root} (checked one level "
            "deep). Run scripts/inspect_schemas.py to see the actual layout."
        )
    return matches[0]


def load_waveform(root: Path, filename_lr: str) -> np.ndarray:
    """filename_lr e.g. 'records100/00000/00001_lr' (no extension). `root` must
    already be resolved via resolve_signal_root — returns [n_leads, n_samples]."""
    record = wfdb.rdrecord(str(root / filename_lr))
    return record.p_signal.T.astype(np.float32)  # [n_leads, n_samples]


def fit_normalization(train_df: pd.DataFrame, cols: list) -> dict:
    """Mean/std computed on the TRAIN split only (avoids leaking val/test
    statistics into the normalization applied everywhere). Some ECGdeli
    features are NaN whenever a wave isn't detected in a given record (e.g.
    no P wave found) — on a small train slice a column can end up ALL-NaN,
    making its mean NaN too, so a bare fillna(mean) would leave NaN in place.
    Falling back to 0.0 (neutral after z-scoring) keeps every column finite."""
    means = train_df[cols].mean().fillna(0.0)
    stds = train_df[cols].std().replace(0, 1.0).fillna(1.0)
    return {"mean": means, "std": stds, "cols": cols}


def apply_normalization(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    out = df.copy()
    out[stats["cols"]] = df[stats["cols"]].fillna(stats["mean"])
    out[stats["cols"]] = (out[stats["cols"]] - stats["mean"]) / stats["std"]
    return out


def official_fold_split(database: pd.DataFrame, val_fold: int = 9, test_fold: int = 10):
    """PTB-XL's own recommended split via `strat_fold` (1-10): fold 9 = val,
    fold 10 = test, 1-8 = train — this is the shipped stratified split, not
    one we construct ourselves."""
    train_ids = database.index[~database["strat_fold"].isin([val_fold, test_fold])]
    val_ids = database.index[database["strat_fold"] == val_fold]
    test_ids = database.index[database["strat_fold"] == test_fold]
    return train_ids, val_ids, test_ids


class PTBXLDataset(Dataset):
    def __init__(self, ecg_ids, database: pd.DataFrame, labels: pd.DataFrame,
                 features: pd.DataFrame, signal_root: Path = PTBXL_SIGNAL_DIR):
        self.ecg_ids = list(ecg_ids)
        self.database = database
        self.labels = labels
        self.features = features
        self.signal_root = resolve_signal_root(Path(signal_root))
        self.feature_cols = [c for c in features.columns if c not in DEMOGRAPHIC_COLS]

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int) -> dict:
        ecg_id = self.ecg_ids[idx]
        row = self.database.loc[ecg_id]
        signal = load_waveform(self.signal_root, row["filename_lr"])
        feat_row = self.features.loc[ecg_id]
        demographics = feat_row[DEMOGRAPHIC_COLS].to_numpy(dtype=np.float32)
        features = feat_row[self.feature_cols].to_numpy(dtype=np.float32)
        label = self.labels.loc[ecg_id].to_numpy(dtype=np.float32)
        return dict(signal=signal, features=features, demographics=demographics, label=label)
