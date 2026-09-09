#!/usr/bin/env bash
# Downloads the four Kaggle datasets (three for the trimodal model, one two-part
# for the ECG experiment) into ../data, matching the layout every loader in
# src/data expects. Requires `kaggle` installed and a valid kaggle.json already
# in place (see README.md for the exact path on Windows).
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root

if ! command -v kaggle >/dev/null 2>&1; then
    echo "kaggle CLI not found. Install with: pip install kaggle" >&2
    exit 1
fi

mkdir -p data/stream1_ehr data/stream2_notes data/stream3_symptoms data/ptbxl_signal data/ptbxl_features

echo "== Stream 1: PhysioNet-2012 ICU mortality =="
kaggle datasets download -d msafi04/predict-mortality-of-icu-patients-physionet -p ./data/stream1_ehr --unzip

echo "== Stream 2: Medical transcriptions (mtsamples) =="
kaggle datasets download -d tboyle10/medicaltranscriptions -p ./data/stream2_notes --unzip

echo "== Stream 3: Symptom2disease =="
kaggle datasets download -d niyarrbarman/symptom2disease -p ./data/stream3_symptoms --unzip

echo "== PTB-XL raw signal =="
kaggle datasets download -d khyeh0719/ptb-xl-dataset -p ./data/ptbxl_signal --unzip

echo "== PTB-XL+ features/demographics =="
kaggle datasets download -d antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset -p ./data/ptbxl_features --unzip

echo "Done. Run: python cardiosentinel/scripts/inspect_schemas.py"
