# CardioSentinel

Two independent experiments:

1. **Trimodal deterioration model** (`src/train_trimodal.py`) — predicts P(deterioration) at 24h/48h/72h horizons by fusing ICU vitals/labs (Stream 1), clinical notes (Stream 2), and patient-reported symptoms (Stream 3).
2. **ECG risk classifier** (`src/train_ecg.py`) — predicts PTB-XL diagnostic superclasses (NORM/MI/STTC/CD/HYP) from ECG signal + engineered features + demographics. Reported separately; not fused into (1).

## ⚠️ Critical limitation: Experiment 1 uses synthetically paired data

**Streams 1, 2, and 3 do not share patients.** PhysioNet-2012 (ICU stays), mtsamples (clinical notes), and Symptom2disease (symptom descriptions) are three unrelated public datasets with no real join key. There is no way to link a specific ICU stay to a specific clinical note or symptom description in reality.

To let the trimodal architecture exercise all three encoders jointly, `src/data/synthetic_pairing.py` **artificially pairs** each (ICU stay, hour) row with a sampled note and a sampled symptom description. The sampling is *not* uniform: it's biased by the row's own deterioration label — rows about to deteriorate are preferentially paired with text that reads as more urgent/distressed (via a lexical keyword-count heuristic, `acuity_score`), and stable rows with more routine-reading text. This creates learnable cross-modal correlation for the model to exploit, but **that correlation is an artifact of the pairing procedure, not a real clinical relationship**.

Concretely, this means:
- Any reported gain from adding Stream 2 / Stream 3 to Stream 1 reflects how well the model learned the synthetic pairing rule, not a validated real-world multimodal signal.
- The "missing-modality robustness" and AMT-gate results are about the *architecture's* behavior, not evidence that real clinical notes/symptoms would carry this information for real ICU patients.
- **Every joint row carries `is_synthetic_pairing=True` and a `pairing_seed` column** for exactly this reason — so no downstream consumer of this data can mistake it for real linkage, and pairing is fully reproducible.
- The pairing-seed sensitivity ablation (retraining with `--pairing-seed 1337` vs the default 42) exists specifically to show how much of the model's apparent performance is sensitive to the arbitrary pairing draw, as a proxy for how much of it is an artifact.

This should be stated explicitly in any paper's methodology and limitations sections: Experiment 1's multimodal results are a **proof-of-concept for the fusion architecture**, not a validated clinical finding, until real trimodal patient-linked data is available.

Experiment 2 (ECG) has no such issue — PTB-XL signal, features, and demographics are the same patients, joined on `ecg_id`.

## Environment this was built against

No GPU was available in the dev environment (CPU-only PyTorch, ~8GB RAM). Given that, this repo is built to **target a GPU** (real ClinicalBERT/BioBERT fine-tuning, full batch sizes — `configs/trimodal.yaml`, `configs/ecg.yaml`), with a CPU-only smoke path to prove correctness without one:

- `scripts/smoke_test.py` — pure architecture check (random tensors, a tiny randomly-initialized text encoder instead of downloading real BERT weights). Runs in under a second on any CPU. Proves shapes, gradient flow, the missing-modality ablation, and the fixed-fusion baseline all work.
- `configs/trimodal_smoke.yaml` / `configs/ecg_smoke.yaml` — real data, real (frozen, not fine-tuned) pretrained encoders, tiny hidden dims, a subsampled number of ICU stays/ECG records. Slow-ish but CPU-feasible; validates the full pipeline (schema → labels → pairing → dataset → training loop) end-to-end. Not meant to produce a usable model.

Move to a GPU machine and use `configs/trimodal.yaml` / `configs/ecg.yaml` for real results.

## Setup

```bash
pip install -r requirements.txt
```

### Data

Get a Kaggle API credential (classic `kaggle.json` via Kaggle → Account → "Create New API Token", or a newer per-account API token) and either:
- place `kaggle.json` at `~/.kaggle/kaggle.json` and run `scripts/download_data.sh`, or
- use `scripts/kaggle_download.py`, which authenticates via a Bearer token read from `~/.kaggle/access_token` — needed because, as of `kaggle` 1.7.4.5 on PyPI, the CLI only supports the classic `kaggle.json` username/key pair, not newer per-account tokens.

Then **always run schema inspection before trusting any loader**:

```bash
python scripts/inspect_schemas.py
```

Loaders in `src/data/` hard-code column names as constants at the top of each file, verified against this project's actual downloaded copies of these datasets. If a different copy of a dataset has different columns, every loader fails fast with the actual columns found rather than silently mis-parsing — update the constants named in the error.

## Running

All commands run from inside `cardiosentinel/` (so `src` resolves as a package):

```bash
# Experiment 1
python -m src.train_trimodal --config configs/trimodal_smoke.yaml   # CPU correctness run
python -m src.train_trimodal --config configs/trimodal.yaml         # real GPU run
scripts/run_ablations.sh configs/trimodal.yaml                       # full ablation suite

# Experiment 2
python -m src.train_ecg --config configs/ecg_smoke.yaml
python -m src.train_ecg --config configs/ecg.yaml
```

## Experiment 1 details

**Labeling** (`src/data/labeling.py`): for each ICU stay, the deterioration event is the earlier of (a) in-hospital death, at `survival_days * 24` hours (falling back to `length_of_stay_days * 24` when `survival_days` is unusable — PhysioNet-2012 encodes `-1` for most survivors), or (b) a vitals-instability trend: HR > 120 or systolic BP < 90, sustained for >= 2 consecutive hours. For every (stay, hour=t) with no event yet, `deteriorate_{24h,48h,72h}` = 1 if the event falls within that many hours forward of t. Rows at/after the event are dropped. On the full downloaded set-a cohort (3,997 stays, ~1.6M raw readings): 13.85% in-hospital mortality, and 11.0% / 13.8% / 14.5% positive rates for the 24h/48h/72h labels respectively — consistent with the "rare event, AUPRC matters" framing in the spec.

**Architecture**: FT-Transformer per ICU hour (feature tokenizer with an additive per-feature missingness embedding, not just zero-imputation) → attention-pooled across hours → projected alongside pooled ClinicalBERT/BioBERT `[CLS]` embeddings → a 3-token self-attention block (cross-modal attention) → an Adaptive Modality-Trust gate that softmaxes per-sample weights over the three streams, conditioned on an explicit presence flag so a genuinely-missing stream always gets exactly zero weight → three sigmoid heads.

**Causality**: a training row for hour t only ever sees EHR data from hours `<= t` (`ehr_time_valid_mask` zeroes out the rest before temporal pooling) — hours after t never influence the prediction for that row.

**Split**: by `stay_id`, not by row, so no timestep of a given stay crosses train/val/test.

## Experiment 2 details

Three branches over the same PTB-XL/PTB-XL+ patients (joined on `ecg_id`): a small 1D-ResNet over the raw 12-lead waveform, an MLP over PTB-XL+'s engineered features, and a separate small MLP over demographics (age/sex/height/weight) — kept apart from the feature branch for a clean per-branch ablation. Concatenated → dense layers → multi-label sigmoid over the 5 diagnostic superclasses. Uses PTB-XL's own shipped `strat_fold` (fold 9 = val, fold 10 = test by default) rather than a custom split. Feature/demographic normalization stats are fit on the train fold only and applied to val/test, to avoid leakage.

**Correction from inspection**: PTB-XL+ actually ships three separate ECG feature-extraction algorithms' outputs (12SL commercial, ECGdeli open-source, UniG) — no demographics bundled in, despite the dataset's description. `src/data/ptbxl_dataset.py` uses ECGdeli (open/reproducible, not a commercial black box) for the feature branch; all demographics come from `ptbxl_database.csv` instead, where height/weight are heavily missing (~68%/57% NaN) — imputed with the train-set mean, so the demographic branch effectively sees "average" for most patients on those two fields.
