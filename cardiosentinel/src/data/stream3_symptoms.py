"""
Stream 3 loader: Symptom2disease (niyarrbarman/symptom2disease).

Documented schema: Unnamed: 0, label (disease name), text (symptom narrative).
Schema is printed on every load; a missing expected column raises immediately
naming what was actually found.
"""
from pathlib import Path

import pandas as pd

from ..paths import STREAM3_DIR

EXPECTED_COLUMNS = ["label", "text"]


def load_stream3_raw(root: Path = STREAM3_DIR) -> pd.DataFrame:
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found under {root}. Run scripts/download_data.sh first.")
    df = pd.read_csv(csvs[0])

    print(f"[stream3_symptoms] loaded {csvs[0].name}: shape={df.shape}, columns={list(df.columns)}")

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Symptom2disease CSV missing expected columns {missing}; found {list(df.columns)}. "
            "Update EXPECTED_COLUMNS in src/data/stream3_symptoms.py."
        )
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df
