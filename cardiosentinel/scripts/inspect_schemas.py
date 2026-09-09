"""
Schema inspection — run this BEFORE trusting any column name in src/data/*.

Prints shape/columns/dtypes/head for every CSV found under each data/ subfolder,
plus a directory listing for the PTB-XL signal folder (which is mostly .dat/.hea
waveform files rather than CSVs). Loaders in src/data hard-code column-name
constants derived from this output; if a dataset's schema doesn't match what's
documented in the download guide, this is where that becomes visible.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.paths import (
    DATA_ROOT, STREAM1_DIR, STREAM2_DIR, STREAM3_DIR,
    PTBXL_SIGNAL_DIR, PTBXL_FEATURES_DIR,
)


def inspect_csv(path: Path, n_head: int = 3) -> None:
    try:
        df = pd.read_csv(path, nrows=5000)
    except Exception as exc:
        print(f"    !! failed to read as CSV: {exc}")
        return
    print(f"    shape (first 5000 rows): {df.shape}")
    print(f"    columns: {list(df.columns)}")
    print(f"    dtypes:\n{df.dtypes.to_string(max_rows=40)}")
    print(f"    head:\n{df.head(n_head).to_string()}")


def inspect_dir(name: str, directory: Path, max_csv: int = 5) -> None:
    print(f"\n{'=' * 70}\n{name}: {directory}\n{'=' * 70}")
    if not directory.exists() or not any(directory.iterdir()):
        print("  NOT DOWNLOADED YET (directory missing or empty). "
              "Run scripts/download_data.sh first.")
        return

    all_files = sorted(p for p in directory.rglob("*") if p.is_file())
    print(f"  total files: {len(all_files)}")
    by_suffix = {}
    for p in all_files:
        by_suffix.setdefault(p.suffix.lower(), []).append(p)
    for suffix, files in sorted(by_suffix.items(), key=lambda kv: -len(kv[1])):
        print(f"    {suffix or '(no ext)'}: {len(files)} file(s)")

    csvs = by_suffix.get(".csv", [])
    for csv_path in csvs[:max_csv]:
        print(f"\n  -- {csv_path.relative_to(directory)} --")
        inspect_csv(csv_path)
    if len(csvs) > max_csv:
        print(f"\n  ... {len(csvs) - max_csv} more CSV file(s) not shown")

    if not csvs:
        print("\n  no CSVs at top level of suffix scan — sample file paths:")
        for p in all_files[:15]:
            print(f"    {p.relative_to(directory)}")


def main() -> None:
    print(f"DATA_ROOT = {DATA_ROOT}")
    inspect_dir("Stream 1 — PhysioNet-2012 ICU (EHR/vitals)", STREAM1_DIR)
    inspect_dir("Stream 2 — mtsamples clinical notes", STREAM2_DIR)
    inspect_dir("Stream 3 — Symptom2disease", STREAM3_DIR)
    inspect_dir("PTB-XL raw signal", PTBXL_SIGNAL_DIR, max_csv=2)
    inspect_dir("PTB-XL+ features/demographics", PTBXL_FEATURES_DIR)


if __name__ == "__main__":
    main()
