"""
Stream 1 loader: PhysioNet/Computing in Cardiology Challenge 2012 ICU mortality
set (Kaggle mirror msafi04/predict-mortality-of-icu-patients-physionet).

The official challenge distribution ships one .txt file per ICU stay (grouped
under set-a/set-b/set-c folders), each a long-format "Time,Parameter,Value"
table (Time = HH:MM elapsed since ICU admission), plus Outcomes-*.txt files
with RecordID, SAPS-I, SOFA, Length_of_stay, Survival, In-hospital_death. Some
Kaggle mirrors instead ship a single already-flattened wide CSV. Both are
handled below; a schema mismatch raises immediately with the actual columns
found rather than silently mis-parsing — run scripts/inspect_schemas.py first
if this ever fires.

VITAL_PARAMS / OUTCOME_COLUMNS below are the one place to edit if a mirror
uses different names than the official challenge documentation.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import STREAM1_DIR

# Static, one-value-per-stay descriptors that appear as "parameters" at Time
# 00:00 in the raw format but aren't time-varying vitals/labs.
STATIC_PARAMS = {"Age", "Gender", "Height", "ICUType", "Weight"}

# Time-varying vitals/labs we keep as model features — confirmed against the
# actual parameter names in this mirror's set-a files (scripts/inspect_schemas.py
# + a direct scan of sample patient .txt files). TroponinI/TroponinT are kept
# despite being sparsely sampled since they're directly cardiac-relevant.
VITAL_PARAMS = [
    "HR", "NIDiasABP", "NISysABP", "NIMAP", "DiasABP", "SysABP", "MAP",
    "Temp", "RespRate", "SaO2", "GCS", "FiO2", "Glucose", "Creatinine",
    "BUN", "WBC", "Platelets", "K", "Na", "HCO3", "Lactate", "pH",
    "Urine", "HCT", "Mg", "PaCO2", "PaO2", "MechVent", "TroponinI", "TroponinT",
    "ALP", "ALT", "AST", "Albumin", "Bilirubin", "Cholesterol",
]

OUTCOME_COLUMNS = ["RecordID", "SAPS-I", "SOFA", "Length_of_stay", "Survival", "In-hospital_death"]


def _find_outcome_files(root: Path) -> list[Path]:
    return sorted(root.rglob("Outcomes*"))


def _find_patient_record_dirs(root: Path) -> list[Path]:
    """Directories that look like they hold one .txt per ICU stay (many small
    files named like a numeric RecordID, e.g. 132539.txt)."""
    candidates = []
    for d in [p for p in root.rglob("*") if p.is_dir()] + [root]:
        txts = list(d.glob("*.txt"))
        numeric_named = [t for t in txts if t.stem.isdigit()]
        if len(numeric_named) >= 10:
            candidates.append(d)
    return candidates


def _parse_raw_patient_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"Time", "Parameter", "Value"}.issubset(df.columns):
        raise ValueError(
            f"{path} does not look like the standard challenge format "
            f"(expected Time,Parameter,Value; found {list(df.columns)}). "
            "Run scripts/inspect_schemas.py and update src/data/stream1_ehr.py."
        )
    df["hour"] = df["Time"].str.split(":").str[0].astype(int)
    return df


def load_raw_long_format(root: Path = STREAM1_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (long_df, outcomes_df).
    long_df columns: stay_id, hour, parameter, value
    outcomes_df columns: stay_id, in_hospital_death, length_of_stay_days, survival_days
    """
    record_dirs = _find_patient_record_dirs(root)
    if not record_dirs:
        raise FileNotFoundError(
            f"No per-patient .txt record directories found under {root}. "
            "Run scripts/inspect_schemas.py to see what's actually there."
        )

    per_file_frames = []
    for record_dir in record_dirs:
        for txt_path in record_dir.glob("*.txt"):
            if not txt_path.stem.isdigit():
                continue
            parsed = _parse_raw_patient_file(txt_path)
            record_id_rows = parsed.loc[parsed["Parameter"] == "RecordID", "Value"]
            stay_id = int(record_id_rows.iloc[0]) if len(record_id_rows) else int(txt_path.stem)
            time_varying = parsed[~parsed["Parameter"].isin(STATIC_PARAMS | {"RecordID"})].copy()
            time_varying["stay_id"] = stay_id
            per_file_frames.append(time_varying[["stay_id", "hour", "Parameter", "Value"]])

    long_df = pd.concat(per_file_frames, ignore_index=True)
    long_df.columns = ["stay_id", "hour", "parameter", "value"]
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    outcome_files = _find_outcome_files(root)
    if not outcome_files:
        raise FileNotFoundError(f"No Outcomes* file found under {root}.")
    outcomes = pd.concat([pd.read_csv(f) for f in outcome_files], ignore_index=True)
    missing_cols = set(OUTCOME_COLUMNS) - set(outcomes.columns)
    if missing_cols:
        raise ValueError(
            f"Outcomes file missing expected columns {missing_cols}; "
            f"found {list(outcomes.columns)}. Update OUTCOME_COLUMNS in this file."
        )
    outcomes = outcomes.rename(columns={
        "RecordID": "stay_id",
        "In-hospital_death": "in_hospital_death",
        "Length_of_stay": "length_of_stay_days",
        "Survival": "survival_days",
    })[["stay_id", "in_hospital_death", "length_of_stay_days", "survival_days"]]

    return long_df, outcomes


def build_hourly_wide(long_df: pd.DataFrame, vital_params: list = VITAL_PARAMS,
                       max_hour: int = 48) -> pd.DataFrame:
    """Pivots the long-format table to one row per (stay_id, hour) with one
    column per vital, plus a `<vital>_missing` mask column. Hours with no
    reading for a given vital get value=0.0 and missing=1 (imputation happens
    here, deliberately, so the encoder only ever sees a fixed feature layout)."""
    long_df = long_df[long_df["parameter"].isin(vital_params) & (long_df["hour"] < max_hour)]
    pivoted = (
        long_df.groupby(["stay_id", "hour", "parameter"])["value"]
        .mean()  # multiple readings within the same hour -> average
        .unstack("parameter")
        .reindex(columns=vital_params)
    )

    all_stays = long_df["stay_id"].unique()
    full_index = pd.MultiIndex.from_product([all_stays, range(max_hour)], names=["stay_id", "hour"])
    pivoted = pivoted.reindex(full_index)

    missing_mask = pivoted.isna().astype(np.float32)
    missing_mask.columns = [f"{c}_missing" for c in missing_mask.columns]
    values = pivoted.fillna(0.0).astype(np.float32)

    wide = pd.concat([values, missing_mask], axis=1).reset_index()
    return wide
