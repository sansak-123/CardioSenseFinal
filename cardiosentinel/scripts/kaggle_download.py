"""
Downloads Kaggle datasets via the raw REST API using a Bearer access token
(~/.kaggle/access_token) instead of the `kaggle` CLI — the CLI package on
PyPI (as of 1.7.4.5) only supports the older kaggle.json username/key pair,
not Kaggle's newer per-account API tokens.
"""
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

TOKEN_PATH = Path.home() / ".kaggle" / "access_token"
API_BASE = "https://www.kaggle.com/api/v1"


def load_token() -> str:
    if not TOKEN_PATH.exists():
        raise FileNotFoundError(f"No token at {TOKEN_PATH}")
    return TOKEN_PATH.read_text().strip()


def download_dataset(ref: str, dest_dir: Path) -> None:
    """ref like 'owner/dataset-slug'. Downloads and unzips into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    token = load_token()
    url = f"{API_BASE}/datasets/download/{ref}"
    zip_path = dest_dir / "_download.zip"

    print(f"Downloading {ref} -> {dest_dir}")
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=ref) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()
    print(f"Done: {ref}")


DATASETS = {
    "stream1_ehr": "msafi04/predict-mortality-of-icu-patients-physionet",
    "stream2_notes": "tboyle10/medicaltranscriptions",
    "stream3_symptoms": "niyarrbarman/symptom2disease",
    "ptbxl_signal": "khyeh0719/ptb-xl-dataset",
    "ptbxl_features": "antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset",
}

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.paths import DATA_ROOT

    targets = sys.argv[1:] or list(DATASETS.keys())
    for name in targets:
        download_dataset(DATASETS[name], DATA_ROOT / name)
