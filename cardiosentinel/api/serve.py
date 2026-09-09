"""
Real-inference API for the CardioSentinel demo — loads whichever trained
checkpoints exist (see model_registry.py) and serves live predictions.

Run from inside cardiosentinel/:
    uvicorn api.serve:app --port 8000 --reload

Two endpoints, one per (independent) experiment:
  POST /predict/trimodal — current vitals + note text + symptom text -> 24h/48h/72h risk
  POST /predict/ecg      — 12-lead signal + demographics -> diagnostic superclass scores
  GET  /health           — which checkpoints (if any) are actually loaded

Trimodal input note: a brand-new patient won't have 48 hours of history, so the
single current-hour reading is placed at hour=0 with `time_valid_mask` marking
only that hour as observed — this is exactly the causal "only see hours <= t"
design already built for training, not a special case bolted on for the API.

ECG input note: the 531 engineered features come from ECGdeli, a third-party
signal-processing toolbox not integrated here — live requests get a
population-average (zero, post z-scoring) feature vector rather than a real
extraction. Only the signal and demographics are genuinely live for this branch.
"""
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .model_registry import (
    load_trimodal_model, load_ecg_model, N_ECG_FEATURES, DEMOGRAPHIC_STATS,
)
from src.data.stream1_ehr import VITAL_PARAMS
from src.data.trimodal_dataset import HORIZON_NAMES
from src.models.ecg_model import PTBXL_SUPERCLASSES

app = FastAPI(title="CardioSentinel Inference API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HOUR = 48


class TrimodalPredictRequest(BaseModel):
    vitals: dict[str, float] = Field(default_factory=dict, description=f"Any subset of: {VITAL_PARAMS}")
    note_text: str = ""
    symptom_text: str = ""


class TrimodalPredictResponse(BaseModel):
    risk: dict[str, float]
    gate_weights: dict[str, float]
    checkpoint: str
    is_smoke_checkpoint: bool
    warnings: list[str]


class ECGPredictRequest(BaseModel):
    age: float
    sex: int = Field(ge=0, le=1)
    height: Optional[float] = None
    weight: Optional[float] = None
    signal: list[list[float]] = Field(description="[12 leads][1000 samples], 100Hz/10s")


class ECGPredictResponse(BaseModel):
    classes: dict[str, float]
    checkpoint: str
    is_smoke_checkpoint: bool
    warnings: list[str]


@app.get("/health")
def health():
    trimodal_model, trimodal_ckpt, trimodal_smoke = load_trimodal_model()
    ecg_model, ecg_ckpt, ecg_smoke = load_ecg_model()
    return {
        "trimodal": {
            "available": trimodal_model is not None,
            "checkpoint": str(trimodal_ckpt) if trimodal_ckpt else None,
            "is_smoke_checkpoint": trimodal_smoke,
        },
        "ecg": {
            "available": ecg_model is not None,
            "checkpoint": str(ecg_ckpt) if ecg_ckpt else None,
            "is_smoke_checkpoint": ecg_smoke,
        },
    }


@app.post("/predict/trimodal", response_model=TrimodalPredictResponse)
def predict_trimodal(req: TrimodalPredictRequest):
    model, ckpt_path, is_smoke = load_trimodal_model()
    if model is None:
        raise HTTPException(503, "No trimodal checkpoint found yet — train one first "
                                  "(python -m src.train_trimodal --config configs/...).")

    warnings = []
    unknown = set(req.vitals) - set(VITAL_PARAMS)
    if unknown:
        warnings.append(f"Ignored unrecognized vitals: {sorted(unknown)}")

    values = torch.zeros(1, MAX_HOUR, len(VITAL_PARAMS))
    missing_mask = torch.ones(1, MAX_HOUR, len(VITAL_PARAMS))
    for i, name in enumerate(VITAL_PARAMS):
        if name in req.vitals:
            values[0, 0, i] = req.vitals[name]
            missing_mask[0, 0, i] = 0.0
    time_valid_mask = torch.zeros(1, MAX_HOUR)
    time_valid_mask[0, 0] = 1.0

    note_text = req.note_text.strip()
    symptom_text = req.symptom_text.strip()
    presence = torch.ones(1, 3)
    if not req.vitals:
        presence[0, 0] = 0.0
        warnings.append("No vitals provided — EHR stream treated as absent.")
    if not note_text:
        presence[0, 1] = 0.0
        note_text = "no note provided"
    if not symptom_text:
        presence[0, 2] = 0.0
        symptom_text = "no symptoms provided"

    notes_enc = model.stream2_encoder.tokenize([note_text], device=torch.device("cpu"))
    symptoms_enc = model.stream3_encoder.tokenize([symptom_text], device=torch.device("cpu"))

    with torch.no_grad():
        out = model(
            values, missing_mask, time_valid_mask,
            notes_enc["input_ids"], notes_enc["attention_mask"],
            symptoms_enc["input_ids"], symptoms_enc["attention_mask"],
            presence,
        )

    risk = {h: torch.sigmoid(out[h]).item() for h in HORIZON_NAMES}
    gate = out["gate_weights"][0].tolist()
    _add_smoke_warning(is_smoke, warnings)

    return TrimodalPredictResponse(
        risk=risk,
        gate_weights={"vitals": gate[0], "notes": gate[1], "symptoms": gate[2]},
        checkpoint=str(ckpt_path.relative_to(ckpt_path.parents[2])),
        is_smoke_checkpoint=is_smoke,
        warnings=warnings,
    )


@app.post("/predict/ecg", response_model=ECGPredictResponse)
def predict_ecg(req: ECGPredictRequest):
    model, ckpt_path, is_smoke = load_ecg_model()
    if model is None:
        raise HTTPException(503, "No ECG checkpoint found yet — train one first "
                                  "(python -m src.train_ecg --config configs/...).")

    warnings = ["Engineered-feature branch uses population-average values — ECGdeli "
                "extraction isn't run on live uploads, only signal + demographics are real."]
    if len(req.signal) != 12:
        raise HTTPException(422, f"Expected 12 leads, got {len(req.signal)}.")
    lengths = {len(lead) for lead in req.signal}
    if lengths != {1000}:
        raise HTTPException(422, f"Expected 1000 samples per lead (100Hz/10s), got lengths {lengths}.")

    signal = torch.tensor(req.signal, dtype=torch.float32).unsqueeze(0)  # [1, 12, 1000]
    features = torch.zeros(1, N_ECG_FEATURES)  # population average, post z-scoring

    demo_values = {"age": req.age, "sex": float(req.sex), "height": req.height, "weight": req.weight}
    demo_norm = []
    for col in ("age", "sex", "height", "weight"):
        v = demo_values[col]
        stats = DEMOGRAPHIC_STATS[col]
        if v is None:
            demo_norm.append(0.0)
            warnings.append(f"'{col}' not provided — used population average.")
        else:
            demo_norm.append((v - stats["mean"]) / stats["std"])
    demographics = torch.tensor([demo_norm], dtype=torch.float32)

    with torch.no_grad():
        logits = model(signal, features, demographics)
    scores = torch.sigmoid(logits)[0].tolist()
    _add_smoke_warning(is_smoke, warnings)

    return ECGPredictResponse(
        classes=dict(zip(PTBXL_SUPERCLASSES, scores)),
        checkpoint=str(ckpt_path.relative_to(ckpt_path.parents[2])),
        is_smoke_checkpoint=is_smoke,
        warnings=warnings,
    )


def _add_smoke_warning(is_smoke: bool, warnings: list) -> None:
    if is_smoke:
        warnings.append("Using the SMOKE checkpoint (tiny dims, frozen encoders, tiny data "
                         "subset) — predictions are not meaningful, this is a correctness check only.")
