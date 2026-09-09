import React, { useState } from "react";
import { getRiskColor } from "@/utils/risk";

const API_BASE = "http://localhost:8000";
const COMMON_VITALS = ["HR", "NISysABP", "NIDiasABP", "Temp", "RespRate", "SaO2", "GCS"];

function pct(x) {
  return `${Math.round(x * 100)}%`;
}

function generateSyntheticSignal() {
  // Placeholder waveform for demo purposes only — NOT a real/clinical ECG.
  const leads = [];
  for (let l = 0; l < 12; l++) {
    const lead = [];
    for (let i = 0; i < 1000; i++) {
      const t = i / 100;
      const beat = Math.sin(2 * Math.PI * 1.2 * t) * (l % 3 === 0 ? 1 : 0.6);
      const noise = (Math.random() - 0.5) * 0.05;
      lead.push(+(beat + noise).toFixed(4));
    }
    leads.push(lead);
  }
  return leads;
}

function parseSignalCSV(text) {
  return text.trim().split("\n").map((row) => row.split(",").map(Number));
}

function Field({ label, children }) {
  return (
    <label className="text-xs text-text-muted">
      {label}
      {children}
    </label>
  );
}

const inputCls = "mt-1 w-full rounded border border-bg-border bg-bg-elevated px-2 py-1.5 text-sm text-text-primary";

function WarningList({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <div className="mt-2 space-y-1">
      {warnings.map((w, i) => (
        <p key={i} className="text-xs text-warning">⚠ {w}</p>
      ))}
    </div>
  );
}

export default function NewPatient() {
  const [vitals, setVitals] = useState({});
  const [noteText, setNoteText] = useState("");
  const [symptomText, setSymptomText] = useState("");
  const [trimodalResult, setTrimodalResult] = useState(null);
  const [trimodalLoading, setTrimodalLoading] = useState(false);
  const [trimodalError, setTrimodalError] = useState(null);

  const [age, setAge] = useState("");
  const [sex, setSex] = useState("0");
  const [height, setHeight] = useState("");
  const [weight, setWeight] = useState("");
  const [signal, setSignal] = useState(null);
  const [ecgResult, setEcgResult] = useState(null);
  const [ecgLoading, setEcgLoading] = useState(false);
  const [ecgError, setEcgError] = useState(null);

  const setVital = (name, value) => {
    setVitals((v) => {
      const next = { ...v };
      if (value === "") delete next[name];
      else next[name] = Number(value);
      return next;
    });
  };

  const submitTrimodal = async () => {
    setTrimodalLoading(true);
    setTrimodalError(null);
    setTrimodalResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict/trimodal`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vitals, note_text: noteText, symptom_text: symptomText }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      setTrimodalResult(body);
    } catch (e) {
      setTrimodalError(
        e.message === "Failed to fetch"
          ? "Can't reach the inference API — is it running (uvicorn api.serve:app --port 8000)?"
          : e.message
      );
    } finally {
      setTrimodalLoading(false);
    }
  };

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const rows = parseSignalCSV(reader.result);
        if (rows.length !== 12) throw new Error(`Expected 12 rows (leads), got ${rows.length}`);
        setSignal(rows);
        setEcgError(null);
      } catch (err) {
        setEcgError(`Couldn't parse file: ${err.message}`);
      }
    };
    reader.readAsText(file);
  };

  const submitEcg = async () => {
    if (!signal) {
      setEcgError("Provide a signal first — upload a CSV or generate a demo one.");
      return;
    }
    setEcgLoading(true);
    setEcgError(null);
    setEcgResult(null);
    try {
      const res = await fetch(`${API_BASE}/predict/ecg`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          age: Number(age),
          sex: Number(sex),
          height: height ? Number(height) : null,
          weight: weight ? Number(weight) : null,
          signal,
        }),
      });
      const body = await res.json();
      if (!res.ok) throw new Error(body.detail || res.statusText);
      setEcgResult(body);
    } catch (e) {
      setEcgError(
        e.message === "Failed to fetch"
          ? "Can't reach the inference API — is it running (uvicorn api.serve:app --port 8000)?"
          : e.message
      );
    } finally {
      setEcgLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-5xl mx-auto">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">New Patient — Live Prediction</h1>
        <p className="text-sm text-text-secondary mt-1">
          Calls the actual trained checkpoints through a local inference API — not mocked data.
          Results may currently come from an early/smoke checkpoint; check the warnings under each result.
        </p>
      </div>

      {/* Trimodal intake */}
      <div className="rounded-lg border border-bg-border bg-bg-card p-5">
        <h2 className="text-base font-semibold text-text-primary mb-1">Deterioration risk</h2>
        <p className="text-sm text-text-secondary mb-4">
          Current vitals + clinical note + patient-reported symptoms. Any field can be left blank —
          the model is designed to handle a missing stream, not just a missing value.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          {COMMON_VITALS.map((name) => (
            <Field key={name} label={name}>
              <input
                type="number"
                className={inputCls}
                value={vitals[name] ?? ""}
                onChange={(e) => setVital(name, e.target.value)}
              />
            </Field>
          ))}
        </div>

        <textarea
          className={`${inputCls} mb-3`}
          rows={2}
          placeholder="Clinical note (optional)"
          value={noteText}
          onChange={(e) => setNoteText(e.target.value)}
        />
        <textarea
          className={`${inputCls} mb-3`}
          rows={2}
          placeholder="Patient-reported symptoms (optional)"
          value={symptomText}
          onChange={(e) => setSymptomText(e.target.value)}
        />

        <button
          onClick={submitTrimodal}
          disabled={trimodalLoading}
          className="rounded-lg bg-sky-accent text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {trimodalLoading ? "Running model…" : "Predict"}
        </button>

        {trimodalError && <p className="text-sm text-critical mt-3">{trimodalError}</p>}

        {trimodalResult && (
          <div className="mt-4 pt-4 border-t border-bg-border">
            <div className="grid grid-cols-3 gap-3 mb-3">
              {["24h", "48h", "72h"].map((h) => {
                const frac = trimodalResult.risk[h];
                const color = getRiskColor(frac);
                return (
                  <div key={h} className="rounded-lg border border-bg-border p-3">
                    <p className="text-xs text-text-muted uppercase">{h}</p>
                    <p className="text-2xl font-semibold" style={{ color }}>{pct(frac)}</p>
                  </div>
                );
              })}
            </div>
            <p className="text-xs text-text-muted">
              Gate weights — vitals {pct(trimodalResult.gate_weights.vitals)}, notes{" "}
              {pct(trimodalResult.gate_weights.notes)}, symptoms {pct(trimodalResult.gate_weights.symptoms)}
            </p>
            <WarningList warnings={trimodalResult.warnings} />
          </div>
        )}
      </div>

      {/* ECG intake */}
      <div className="rounded-lg border border-sky-dim/30 bg-bg-elevated p-5">
        <h2 className="text-base font-semibold text-text-primary mb-1">ECG diagnostic classifier</h2>
        <p className="text-sm text-text-secondary mb-4">
          Separate model — 12-lead signal + demographics. Not fused into the prediction above.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <Field label="Age">
            <input type="number" className={inputCls} value={age} onChange={(e) => setAge(e.target.value)} />
          </Field>
          <Field label="Sex">
            <select className={inputCls} value={sex} onChange={(e) => setSex(e.target.value)}>
              <option value="0">Male</option>
              <option value="1">Female</option>
            </select>
          </Field>
          <Field label="Height (cm)">
            <input type="number" className={inputCls} value={height} onChange={(e) => setHeight(e.target.value)} />
          </Field>
          <Field label="Weight (kg)">
            <input type="number" className={inputCls} value={weight} onChange={(e) => setWeight(e.target.value)} />
          </Field>
        </div>

        <div className="flex flex-wrap items-center gap-3 mb-4">
          <input type="file" accept=".csv,.txt" onChange={handleFile} className="text-xs text-text-secondary" />
          <button
            onClick={() => setSignal(generateSyntheticSignal())}
            className="text-xs rounded border border-bg-border px-3 py-1.5 text-text-secondary hover:bg-bg-hover"
          >
            Generate demo signal
          </button>
          {signal && <span className="text-xs text-safe">Signal ready ({signal.length} leads × {signal[0]?.length} samples)</span>}
        </div>
        <p className="text-[11px] text-text-muted mb-4">
          CSV format: 12 rows (one per lead), 1000 comma-separated values each (100Hz, 10s).
          "Generate demo signal" makes a placeholder waveform for testing — it isn't a real ECG.
        </p>

        <button
          onClick={submitEcg}
          disabled={ecgLoading}
          className="rounded-lg bg-sky-dim text-white px-4 py-2 text-sm font-medium disabled:opacity-50"
        >
          {ecgLoading ? "Running model…" : "Classify"}
        </button>

        {ecgError && <p className="text-sm text-critical mt-3">{ecgError}</p>}

        {ecgResult && (
          <div className="mt-4 pt-4 border-t border-bg-border">
            {Object.entries(ecgResult.classes)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, score]) => (
                <div key={cls} className="flex items-center justify-between py-1">
                  <span className="text-sm text-text-secondary">{cls}</span>
                  <span className="text-sm font-medium text-text-primary">{pct(score)}</span>
                </div>
              ))}
            <WarningList warnings={ecgResult.warnings} />
          </div>
        )}
      </div>
    </div>
  );
}
