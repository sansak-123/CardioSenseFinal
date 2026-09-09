"""
Synthetic cross-dataset pairing for Experiment 1.

Streams 1 (PhysioNet-2012 ICU stays), 2 (mtsamples cardiac notes), and 3
(Symptom2disease descriptions) do NOT share patients — there is no real join
key. Every joint row produced here is therefore synthetic: a Stream-1
(stay_id, hour) row gets a note sampled from Stream 2 and a symptom
description sampled from Stream 3, biased by the row's own deterioration
label rather than picked uniformly at random, so that "this patient is about
to deteriorate" rows are preferentially paired with distress-indicating text
and stable rows with routine text. This is a modeling convenience to let the
architecture exercise all three encoders jointly — it is NOT a claim that
these texts came from the same patient. See README.md's limitations section.

Every output row carries `is_synthetic_pairing=True` and `pairing_seed` so the
pairing is fully reproducible and impossible to mistake for real linkage.
"""
import re

import numpy as np
import pandas as pd

URGENT_KEYWORDS = [
    "severe", "acute", "sudden", "suddenly", "emergency", "emergent",
    "chest pain", "shortness of breath", "difficulty breathing", "dyspnea",
    "unstable", "critical", "collapse", "syncope", "unresponsive",
    "cardiac arrest", "worsening", "distress", "urgent", "intense",
    "radiating", "diaphoresis", "palpitations", "crushing", "high fever",
    "rapid heart rate", "tachycardia", "hypotension", "bleeding heavily",
]
ROUTINE_KEYWORDS = [
    "routine", "follow-up", "follow up", "stable", "well-controlled",
    "well controlled", "annual", "screening", "mild", "occasional",
    "resolved", "improving", "no acute distress", "unremarkable",
    "as needed", "check-up", "checkup", "minor", "slight",
]

_WORD_RE = re.compile(r"[a-z]+")


def acuity_score(text: str) -> float:
    """Higher = reads as more urgent/distressed. Purely lexical (keyword
    counts normalized by text length) — a heuristic proxy, not a clinical
    acuity score, used only to bias synthetic sampling."""
    lowered = str(text).lower()
    n_words = max(len(_WORD_RE.findall(lowered)), 1)
    urgent = sum(lowered.count(kw) for kw in URGENT_KEYWORDS)
    routine = sum(lowered.count(kw) for kw in ROUTINE_KEYWORDS)
    return 100.0 * (urgent - routine) / n_words


def _sampling_probs(acuity: np.ndarray, favor_high: bool, temperature: float) -> np.ndarray:
    signed = acuity if favor_high else -acuity
    z = signed / max(temperature, 1e-6)
    z = z - z.max()
    weights = np.exp(z)
    return weights / weights.sum()


def build_joint_dataset(stream1_labeled: pd.DataFrame, stream2_notes: pd.DataFrame,
                         stream3_symptoms: pd.DataFrame, pairing_seed: int = 42,
                         temperature: float = 5.0,
                         label_col: str = "any_deterioration") -> pd.DataFrame:
    """
    stream1_labeled: output of labeling.add_deterioration_labels (must have
        deteriorate_24h/48h/72h columns).
    Returns stream1_labeled with added columns: paired_note_text,
    paired_symptom_text, is_synthetic_pairing, pairing_seed.
    """
    df = stream1_labeled.copy()
    df[label_col] = df[["deteriorate_24h", "deteriorate_48h", "deteriorate_72h"]].max(axis=1)

    note_acuity = stream2_notes["transcription"].map(acuity_score).to_numpy()
    symptom_acuity = stream3_symptoms["text"].map(acuity_score).to_numpy()

    rng = np.random.default_rng(pairing_seed)

    probs_note_high = _sampling_probs(note_acuity, favor_high=True, temperature=temperature)
    probs_note_low = _sampling_probs(note_acuity, favor_high=False, temperature=temperature)
    probs_symptom_high = _sampling_probs(symptom_acuity, favor_high=True, temperature=temperature)
    probs_symptom_low = _sampling_probs(symptom_acuity, favor_high=False, temperature=temperature)

    is_positive = df[label_col].to_numpy().astype(bool)
    n = len(df)

    note_idx = np.empty(n, dtype=np.int64)
    symptom_idx = np.empty(n, dtype=np.int64)
    n_pos = int(is_positive.sum())
    n_neg = n - n_pos
    if n_pos:
        note_idx[is_positive] = rng.choice(len(stream2_notes), size=n_pos, p=probs_note_high)
        symptom_idx[is_positive] = rng.choice(len(stream3_symptoms), size=n_pos, p=probs_symptom_high)
    if n_neg:
        note_idx[~is_positive] = rng.choice(len(stream2_notes), size=n_neg, p=probs_note_low)
        symptom_idx[~is_positive] = rng.choice(len(stream3_symptoms), size=n_neg, p=probs_symptom_low)

    df["paired_note_text"] = stream2_notes["transcription"].to_numpy()[note_idx]
    df["paired_symptom_text"] = stream3_symptoms["text"].to_numpy()[symptom_idx]
    df["is_synthetic_pairing"] = True
    df["pairing_seed"] = pairing_seed

    pos_note_acuity = note_acuity[note_idx][is_positive].mean() if n_pos else float("nan")
    neg_note_acuity = note_acuity[note_idx][~is_positive].mean() if n_neg else float("nan")
    print(f"[synthetic_pairing] seed={pairing_seed}: {n} rows paired "
          f"({n_pos} positive, {n_neg} negative). "
          f"mean note acuity — positive rows: {pos_note_acuity:.3f}, "
          f"negative rows: {neg_note_acuity:.3f} "
          f"(higher = more distress-indicating text, by construction)")

    return df
