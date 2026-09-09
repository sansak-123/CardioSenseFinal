#!/usr/bin/env bash
# Runs the full ablation suite for Experiment 1: baseline, each stream removed,
# AMT gate vs fixed-weight fusion, and a second pairing seed for sensitivity.
# Usage: scripts/run_ablations.sh [config_path]   (defaults to configs/trimodal.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."   # cardiosentinel/

CONFIG="${1:-configs/trimodal.yaml}"

echo "== baseline (AMT fusion) =="
python -m src.train_trimodal --config "$CONFIG"

echo "== ablation: stream1 (EHR) removed =="
python -m src.train_trimodal --config "$CONFIG" --ablate-stream 0

echo "== ablation: stream2 (notes) removed =="
python -m src.train_trimodal --config "$CONFIG" --ablate-stream 1

echo "== ablation: stream3 (symptoms) removed =="
python -m src.train_trimodal --config "$CONFIG" --ablate-stream 2

echo "== ablation: fixed-weight fusion baseline (vs AMT gate) =="
python -m src.train_trimodal --config "$CONFIG" --fusion fixed

echo "== ablation: synthetic-pairing sensitivity (alternate seed) =="
python -m src.train_trimodal --config "$CONFIG" --pairing-seed 1337

echo "All ablation runs complete. Checkpoints + logs are per-run; compare the printed TEST metrics."
