"""
Stream 2 loader: mtsamples clinical transcriptions (tboyle10/medicaltranscriptions).

Documented schema: Unnamed: 0, description, medical_specialty, sample_name,
transcription, keywords — medical_specialty values carry a leading space
(e.g. " Cardiovascular / Pulmonary"), so matching is done stripped/lower-cased.
Schema is printed on every load; a missing expected column raises immediately
naming what was actually found, rather than silently returning an empty frame.
"""
from pathlib import Path

import pandas as pd

from ..paths import STREAM2_DIR

EXPECTED_COLUMNS = ["description", "medical_specialty", "sample_name", "transcription", "keywords"]
TARGET_SPECIALTY = "cardiovascular / pulmonary"


def load_stream2_raw(root: Path = STREAM2_DIR) -> pd.DataFrame:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {root}. Run scripts/download_data.sh first.")
    df = pd.read_csv(csvs[0])

    print(f"[stream2_notes] loaded {csvs[0].name}: shape={df.shape}, columns={list(df.columns)}")

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"mtsamples CSV missing expected columns {missing}; found {list(df.columns)}. "
            "Update EXPECTED_COLUMNS in src/data/stream2_notes.py."
        )
    return df


def load_cardiac_notes(root: Path = STREAM2_DIR) -> pd.DataFrame:
    """Filters to the Cardiovascular / Pulmonary specialty and drops rows with
    no transcription text."""
    df = load_stream2_raw(root)
    specialty_norm = df["medical_specialty"].astype(str).str.strip().str.lower()
    cardiac = df[specialty_norm == TARGET_SPECIALTY].copy()
    cardiac = cardiac.dropna(subset=["transcription"])
    cardiac["transcription"] = cardiac["transcription"].astype(str).str.strip()
    cardiac = cardiac[cardiac["transcription"].str.len() > 0].reset_index(drop=True)

    print(f"[stream2_notes] filtered to '{TARGET_SPECIALTY}': {len(cardiac)}/{len(df)} rows")
    if len(cardiac) == 0:
        raise ValueError(
            f"No rows matched specialty '{TARGET_SPECIALTY}'. Actual specialty values: "
            f"{sorted(specialty_norm.unique())[:20]}"
        )
    return cardiac
