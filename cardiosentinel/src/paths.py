"""Central path resolution so every module agrees on where the repo/data live."""
from pathlib import Path

CARDIOSENTINEL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CARDIOSENTINEL_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"

STREAM1_DIR = DATA_ROOT / "stream1_ehr"
STREAM2_DIR = DATA_ROOT / "stream2_notes"
STREAM3_DIR = DATA_ROOT / "stream3_symptoms"
PTBXL_SIGNAL_DIR = DATA_ROOT / "ptbxl_signal"
PTBXL_FEATURES_DIR = DATA_ROOT / "ptbxl_features"

CHECKPOINTS_DIR = CARDIOSENTINEL_ROOT / "checkpoints"
RUNS_DIR = CARDIOSENTINEL_ROOT / "runs"
