import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Search, User, Heart, Activity, Zap, ChevronDown,
  BarChart2, AlertTriangle, FileText, Layers
} from "lucide-react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid
} from "recharts";
import ECGCanvas from "@/components/ECGCanvas";
import { Card, StatCard, RiskBadge, SectionHeader, StreamBadge, Value, Tag, Button } from "@/components/ui";
import clsx from "clsx";

const PATIENTS = [
  {
    id: "ECG-21401", age: 67, sex: "M", label: "MI", status: "critical",
    risk: { h24: 0.83, h48: 0.88, h72: 0.91 },
    pr: 168, qrs: 112, qt: 440, hr: 94,
    dx: "Anterior ST-elevation myocardial infarction",
    report: "Marked ST elevation in leads V1–V4. Reciprocal ST depression in inferior leads. QRS complex widened. Urgent intervention indicated.",
    shap: [
      { feature: "ST Elevation (V2)", importance: 0.38, stream: "S1" },
      { feature: "QRS Width", importance: 0.24, stream: "S3" },
      { feature: '"urgent" in report', importance: 0.18, stream: "S2" },
      { feature: "HR trend (+12%)", importance: 0.12, stream: "S1" },
      { feature: "QT prolongation", importance: 0.08, stream: "S3" },
    ],
    trust: [{ m: "ECG Signal", v: 0.62 }, { m: "Clinical Text", v: 0.28 }, { m: "Structured", v: 0.10 }],
  },
  {
    id: "ECG-18834", age: 54, sex: "F", label: "STTC", status: "warning",
    risk: { h24: 0.61, h48: 0.68, h72: 0.74 },
    pr: 152, qrs: 88, qt: 420, hr: 78,
    dx: "ST/T-wave changes — possible ischaemia",
    report: "Non-specific ST-T changes in lateral leads. T-wave inversion V4-V6. Clinical correlation recommended.",
    shap: [
      { feature: "T-wave inversion (V5)", importance: 0.31, stream: "S1" },
      { feature: '"T-wave inversion" text', importance: 0.29, stream: "S2" },
      { feature: "QT interval", importance: 0.20, stream: "S3" },
      { feature: "HR variability", importance: 0.13, stream: "S1" },
      { feature: "Age 54F", importance: 0.07, stream: "S3" },
    ],
    trust: [{ m: "ECG Signal", v: 0.44 }, { m: "Clinical Text", v: 0.38 }, { m: "Structured", v: 0.18 }],
  },
  {
    id: "ECG-09271", age: 42, sex: "M", label: "NORM", status: "safe",
    risk: { h24: 0.09, h48: 0.10, h72: 0.11 },
    pr: 148, qrs: 82, qt: 390, hr: 68,
    dx: "Normal sinus rhythm",
    report: "No significant ST or T-wave abnormalities. Normal QRS axis. Normal PR interval. No evidence of hypertrophy.",
    shap: [
      { feature: "Normal P-wave axis", importance: 0.42, stream: "S1" },
      { feature: '"no evidence" in text', importance: 0.25, stream: "S2" },
      { feature: "HR 68 bpm", importance: 0.18, stream: "S3" },
      { feature: "QRS 82ms (normal)", importance: 0.10, stream: "S1" },
      { feature: "PR interval normal", importance: 0.05, stream: "S3" },
    ],
    trust: [{ m: "ECG Signal", v: 0.50 }, { m: "Clinical Text", v: 0.30 }, { m: "Structured", v: 0.20 }],
  },
];

const HORIZON_DATA = (p) => [
  { h: "24h", risk: p.risk.h24 * 100 },
  { h: "48h", risk: p.risk.h48 * 100 },
  { h: "72h", risk: p.risk.h72 * 100 },
];

const STATUS_COLOR = {
  critical: "text-critical bg-critical/10 border-critical/30",
  warning:  "text-warning  bg-warning/10  border-warning/30",
  safe:     "text-safe     bg-safe/10     border-safe/30",
};

export default function PatientAnalysis() {
  const [selected, setSelected] = useState(PATIENTS[0]);
  const [query, setQuery] = useState("");

  const filtered = PATIENTS.filter(p =>
    p.id.toLowerCase().includes(query.toLowerCase()) ||
    p.label.toLowerCase().includes(query.toLowerCase())
  );

  return (
    <div className="flex h-full">
      {/* ── Sidebar list ─────────────────────────────────────────── */}
      <div className="w-60 shrink-0 border-r border-bg-border bg-bg-card flex flex-col">
        <div className="p-3 border-b border-bg-border">
          <div className="flex items-center gap-2 bg-bg-elevated border border-bg-border rounded-lg px-3 py-2">
            <Search size={13} className="text-text-muted" />
            <input
              value={query}
              onChange={e => setQuery(e.target.value)}
              placeholder="Search records…"
              className="bg-transparent text-xs text-text-primary placeholder-text-muted outline-none w-full font-mono"
            />
          </div>
        </div>
        <div className="flex-1 overflow-y-auto py-2">
          {filtered.map(p => (
            <button
              key={p.id}
              onClick={() => setSelected(p)}
              className={clsx(
                "w-full text-left px-3 py-2.5 transition-colors hover:bg-bg-hover mx-0",
                selected.id === p.id && "bg-bg-elevated border-r-2 border-sky-accent"
              )}
            >
              <div className="flex items-center justify-between mb-0.5">
                <span className="text-[10px] font-mono text-text-secondary">{p.id}</span>
                <span className={clsx("text-[8px] font-mono border rounded px-1.5 py-0.5", STATUS_COLOR[p.status])}>
                  {p.label}
                </span>
              </div>
              <p className="text-[10px] text-text-muted">{p.age}y {p.sex} · {Math.round(p.risk.h72 * 100)}% @ 72h</p>
            </button>
          ))}
        </div>
      </div>

      {/* ── Detail panel ─────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto p-5 space-y-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div>
            <p className="text-[10px] font-mono text-sky-accent tracking-widest mb-1">PATIENT RECORD</p>
            <h2 className="font-display text-2xl font-semibold text-text-primary">{selected.id}</h2>
            <p className="text-sm text-text-secondary">{selected.age}y {selected.sex === "M" ? "Male" : "Female"} · {selected.dx}</p>
          </div>
          <span className={clsx("text-xs font-mono border rounded-lg px-3 py-1.5 uppercase tracking-wide", STATUS_COLOR[selected.status])}>
            {selected.status}
          </span>
        </div>

        {/* ECG waveform — colour-coded by risk */}
        <Card className="overflow-hidden p-4">
          <p className="text-[10px] font-mono text-text-muted mb-2 tracking-widest">LIVE 12-LEAD ECG</p>
          <ECGCanvas
            height={90}
            color={selected.status === "critical" ? "#E11D48" : selected.status === "warning" ? "#F59E0B" : "#10B981"}
            risk={selected.risk.h72}
            className="w-full"
          />
          <div className="flex gap-4 mt-2">
            {[["HR", `${selected.hr} bpm`], ["PR", `${selected.pr} ms`], ["QRS", `${selected.qrs} ms`], ["QT", `${selected.qt} ms`]].map(([k, v]) => (
              <div key={k}>
                <p className="text-[9px] font-mono text-text-muted">{k}</p>
                <p className="text-xs font-mono text-text-primary">{v}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Risk horizons */}
        <div className="grid grid-cols-3 gap-3">
          <RiskBadge score={selected.risk.h24} label="24h Risk" />
          <RiskBadge score={selected.risk.h48} label="48h Risk" />
          <RiskBadge score={selected.risk.h72} label="72h Risk" />
        </div>

        {/* Two-col: SHAP + Modality trust */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* SHAP explainability */}
          <Card>
            <SectionHeader eyebrow="Explainability" title="SHAP Feature Attribution" sub="Which signals drove this prediction" />
            <div className="space-y-2">
              {selected.shap.map((s, i) => (
                <div key={i} className="space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <StreamBadge stream={s.stream} />
                      <span className="text-xs text-text-secondary font-mono">{s.feature}</span>
                    </div>
                    <span className="text-xs font-mono text-text-primary">{(s.importance * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-bg-border overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${s.importance * 100}%` }}
                      transition={{ duration: 0.6, delay: i * 0.08 }}
                      className={clsx(
                        "h-full rounded-full",
                        s.stream === "S1" ? "bg-sky-accent" :
                        s.stream === "S2" ? "bg-safe" : "bg-warning"
                      )}
                    />
                  </div>
                </div>
              ))}
            </div>
          </Card>

          {/* Modality trust */}
          <Card>
            <SectionHeader eyebrow="AMT Gate" title="Modality Trust Weights" sub="Dynamic per-step attention" />
            <ResponsiveContainer width="100%" height={160}>
              <RadarChart data={selected.trust.map(t => ({ subject: t.m, value: Math.round(t.v * 100) }))}>
                <PolarGrid stroke="#1E3A5F" />
                <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: "#94A3B8", fontFamily: "JetBrains Mono" }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                <Radar dataKey="value" stroke="#38BDF8" fill="#38BDF8" fillOpacity={0.15} strokeWidth={1.5} />
                <Tooltip formatter={(v) => [`${v}%`, "Trust"]} contentStyle={{ background: "#0A1628", border: "1px solid #1E3A5F", borderRadius: 8, fontSize: 11, fontFamily: "JetBrains Mono" }} />
              </RadarChart>
            </ResponsiveContainer>
            <div className="mt-1 space-y-1">
              {selected.trust.map(t => (
                <div key={t.m} className="flex items-center justify-between">
                  <span className="text-xs text-text-secondary">{t.m}</span>
                  <span className="text-xs font-mono text-text-primary">{Math.round(t.v * 100)}%</span>
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Cardiologist report */}
        <Card>
          <div className="flex items-center gap-2 mb-3">
            <FileText size={14} className="text-sky-accent" />
            <SectionHeader eyebrow="Stream 2 · NLP Input" title="Cardiologist Diagnostic Statement" />
          </div>
          <div className="bg-bg-elevated rounded-lg border border-bg-border px-4 py-3">
            <p className="text-sm text-text-secondary leading-relaxed font-mono">{selected.report}</p>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <Tag color="sky">BioBERT encoded</Tag>
            <Tag color="safe">SNOMED-mapped</Tag>
            <Tag>SCP-ECG standard</Tag>
          </div>
        </Card>
      </div>
    </div>
  );
}