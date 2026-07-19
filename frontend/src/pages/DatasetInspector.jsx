import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Database, ExternalLink, CheckCircle2, Clock,
  FileText, Activity, Layers, ChevronDown, ChevronUp
} from "lucide-react";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from "recharts";
import { Card, SectionHeader, StreamBadge, Tag, Value, Button } from "@/components/ui";
import clsx from "clsx";

const DATASETS = [
  {
    id: "ptbxl",
    name: "PTB-XL",
    version: "v1.0.3",
    status: "primary",
    badge: "REQUIRED",
    badgeColor: "text-sky-accent bg-sky-accent/10 border-sky-accent/30",
    records: 21837, patients: 18885,
    stream: "S1",
    kaggle: "https://www.kaggle.com/datasets/garethwmch/ptb-xl-1-0-3",
    physionet: "https://physionet.org/content/ptb-xl/1.0.3/",
    description: "The gold-standard public ECG dataset. 21,837 clinical 12-lead ECGs of 10-second length from 18,885 patients. Annotated by up to two cardiologists with 71 SCP-ECG diagnostic statements, organised into 5 superclasses.",
    attrs: [
      { name: "ECG waveform (12 leads)", type: "float array",  stream: "S1", note: "100Hz or 500Hz, 10s" },
      { name: "age",                     type: "int · years",  stream: "S3", note: "0–95, median 62" },
      { name: "sex",                     type: "binary",       stream: "S3", note: "52% M, 48% F" },
      { name: "height / weight",         type: "float",        stream: "S3", note: "partially available" },
      { name: "scp_codes",               type: "dict · %",     stream: "S2", note: "NORM, MI, STTC, CD, HYP + likelihood" },
      { name: "report",                  type: "free text",    stream: "S2", note: "cardiologist diagnostic statement" },
      { name: "heart_axis",              type: "categorical",  stream: "S3", note: "electrical axis deviation" },
      { name: "infarction_stadium",      type: "ordinal",      stream: "S3", note: "MI stage if present" },
      { name: "strat_fold",              type: "int 1–10",     stream: "—",  note: "recommended train/val/test split" },
    ],
    dist: [
      { label: "NORM", count: 9528, color: "#10B981" },
      { label: "MI",   count: 5486, color: "#E11D48" },
      { label: "STTC", count: 5250, color: "#F59E0B" },
      { label: "CD",   count: 4907, color: "#38BDF8" },
      { label: "HYP",  count: 2655, color: "#A78BFA" },
    ],
  },
  {
    id: "ptbxlplus",
    name: "PTB-XL+",
    version: "v1.0.1",
    status: "primary",
    badge: "REQUIRED",
    badgeColor: "text-sky-accent bg-sky-accent/10 border-sky-accent/30",
    records: 21837, patients: 18885,
    stream: "S3",
    kaggle: "https://www.kaggle.com/datasets/antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset",
    physionet: "https://physionet.org/content/ptb-xl-plus/1.0.1/",
    description: "Supplementary feature dataset for PTB-XL. Provides ECG intervals and morphological features extracted by three ECG analysis algorithms (Glasgow, Uni-Leipzig, EMREB), plus automatic diagnostic statements from commercial software.",
    attrs: [
      { name: "PR interval",        type: "float · ms",      stream: "S3", note: "Glasgow + Uni-Leipzig + EMREB" },
      { name: "QRS duration",       type: "float · ms",      stream: "S3", note: "three algorithm sources" },
      { name: "QT / QTc interval",  type: "float · ms",      stream: "S3", note: "Bazett-corrected QTc" },
      { name: "Heart rate",         type: "float · bpm",     stream: "S3", note: "from R-R interval" },
      { name: "P-wave axis",        type: "float · degrees", stream: "S3", note: "frontal plane" },
      { name: "QRS axis",           type: "float · degrees", stream: "S3", note: "electrical axis" },
      { name: "T-wave axis",        type: "float · degrees", stream: "S3", note: "repolarisation direction" },
      { name: "ST amplitude (V1–V6)", type: "float · mV",   stream: "S3", note: "ST elevation/depression per lead" },
      { name: "Fiducial points",    type: "array",           stream: "S3", note: "P, Q, R, S, T peak indices" },
      { name: "Auto-diagnosis text", type: "free text",      stream: "S2", note: "commercial ECG interpretation" },
    ],
    dist: [
      { label: "Glasgow",      count: 21837, color: "#38BDF8" },
      { label: "Uni-Leipzig",  count: 21837, color: "#10B981" },
      { label: "EMREB",        count: 18486, color: "#F59E0B" },
    ],
  },
  {
    id: "ptbxl_image",
    name: "PTB-XL-Image-17K",
    version: "2024",
    status: "optional",
    badge: "OPTIONAL",
    badgeColor: "text-warning bg-warning/10 border-warning/30",
    records: 17111, patients: 17111,
    stream: "S1",
    kaggle: null,
    physionet: "https://zenodo.org/records/18197519",
    github: "https://github.com/naqchoalimehdi/PTB-XL-Image-17K",
    description: "ECG paper-style images generated directly from PTB-XL records. Each image shows the 12-lead ECG in standard clinical format, linked to the original signal by ecg_id. Enables optional vision stream or OCR-based text extraction.",
    attrs: [
      { name: "ECG paper image",  type: "PNG 300dpi",   stream: "S1", note: "standard 12-lead grid layout" },
      { name: "ecg_id",           type: "int",          stream: "—",  note: "links to PTB-XL record" },
      { name: "label",            type: "superclass",   stream: "—",  note: "NORM/MI/STTC/CD/HYP" },
      { name: "image_path",       type: "string",       stream: "S1", note: "relative file path" },
    ],
    dist: [
      { label: "NORM", count: 6821, color: "#10B981" },
      { label: "MI",   count: 3814, color: "#E11D48" },
      { label: "STTC", count: 3320, color: "#F59E0B" },
      { label: "CD",   count: 2481, color: "#38BDF8" },
      { label: "HYP",  count: 675,  color: "#A78BFA" },
    ],
  },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-bg-border rounded-lg p-2.5 text-xs font-mono">
      <p className="text-text-muted mb-1">{label}</p>
      <p className="text-text-primary">{payload[0].value.toLocaleString()} records</p>
    </div>
  );
};

export default function DatasetInspector() {
  const [selected, setSelected] = useState("ptbxl");
  const [expandAttrs, setExpandAttrs] = useState(false);
  const ds = DATASETS.find(d => d.id === selected);

  return (
    <div className="p-6 space-y-6">
      <SectionHeader
        eyebrow="Data Sources"
        title="Dataset Inspector"
        sub="All datasets are freely available — no MIMIC access required."
      />

      {/* ── Dataset tabs ─────────────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-3">
        {DATASETS.map(d => (
          <motion.button
            key={d.id}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => { setSelected(d.id); setExpandAttrs(false); }}
            className={clsx(
              "rounded-xl border p-4 text-left transition-all",
              selected === d.id
                ? "border-sky-accent/40 bg-sky-accent/5"
                : "border-bg-border bg-bg-card hover:bg-bg-hover"
            )}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="w-8 h-8 rounded-lg bg-bg-elevated border border-bg-border flex items-center justify-center">
                <Database size={14} className="text-text-secondary" />
              </div>
              <span className={clsx("text-[9px] font-mono border rounded px-2 py-0.5 tracking-wide", d.badgeColor)}>
                {d.badge}
              </span>
            </div>
            <p className="font-display text-sm font-semibold text-text-primary">{d.name}</p>
            <p className="text-[10px] text-text-muted font-mono">{d.version}</p>
            <div className="flex items-center gap-2 mt-2">
              <StreamBadge stream={d.stream} />
            </div>
            <div className="flex gap-3 mt-2">
              <div>
                <p className="text-[9px] text-text-muted">Records</p>
                <p className="text-xs font-mono text-text-primary">{d.records.toLocaleString()}</p>
              </div>
              <div>
                <p className="text-[9px] text-text-muted">Patients</p>
                <p className="text-xs font-mono text-text-primary">{d.patients.toLocaleString()}</p>
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      {/* ── Dataset detail ────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={selected}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.18 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-4"
        >
          {/* Left: description + links */}
          <div className="space-y-4">
            <Card>
              <SectionHeader eyebrow="About" title={ds.name} />
              <p className="text-sm text-text-secondary leading-relaxed">{ds.description}</p>
              <div className="mt-4 space-y-2">
                {ds.kaggle && (
                  <a href={ds.kaggle} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 text-xs text-sky-accent hover:text-sky-glow transition-colors">
                    <ExternalLink size={11} /> Kaggle — Download free
                  </a>
                )}
                {ds.physionet && (
                  <a href={ds.physionet} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 text-xs text-text-secondary hover:text-text-primary transition-colors">
                    <ExternalLink size={11} /> PhysioNet / Zenodo
                  </a>
                )}
                {ds.github && (
                  <a href={ds.github} target="_blank" rel="noopener noreferrer"
                    className="flex items-center gap-2 text-xs text-text-secondary hover:text-text-primary transition-colors">
                    <ExternalLink size={11} /> GitHub repo
                  </a>
                )}
              </div>
            </Card>

            {/* Download steps */}
            <Card>
              <SectionHeader eyebrow="Setup" title="Download Steps" />
              <div className="space-y-2.5">
                {[
                  "Install Kaggle CLI: pip install kaggle",
                  "Get API key: kaggle.com → Settings → API → Create Token",
                  "Place kaggle.json in ~/.kaggle/",
                  ds.id !== "ptbxl_image"
                    ? `Run: python data/download_datasets.py`
                    : "Download from Zenodo link above",
                  "Run: python data/preprocess_ptbxl.py",
                ].map((step, i) => (
                  <div key={i} className="flex items-start gap-2.5">
                    <div className="w-4 h-4 rounded-full bg-sky-accent/10 border border-sky-accent/20 flex items-center justify-center shrink-0 mt-0.5">
                      <span className="text-[8px] font-mono text-sky-accent">{i + 1}</span>
                    </div>
                    <p className="text-xs text-text-secondary font-mono leading-relaxed">{step}</p>
                  </div>
                ))}
              </div>
            </Card>
          </div>

          {/* Middle: distribution chart */}
          <Card>
            <SectionHeader eyebrow="Label Distribution" title="Class Balance" />
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ds.dist} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
                <YAxis tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {ds.dist.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.8} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-2 mt-3">
              {ds.dist.map(d => (
                <div key={d.label} className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-sm" style={{ background: d.color }} />
                  <span className="text-[10px] font-mono text-text-muted">{d.label}: {d.count.toLocaleString()}</span>
                </div>
              ))}
            </div>
          </Card>

          {/* Right: attributes */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <SectionHeader eyebrow="Schema" title={`Attributes (${ds.attrs.length})`} />
              <button
                onClick={() => setExpandAttrs(e => !e)}
                className="text-text-muted hover:text-text-secondary transition-colors"
              >
                {expandAttrs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
              </button>
            </div>
            <div className="space-y-1.5">
              {(expandAttrs ? ds.attrs : ds.attrs.slice(0, 5)).map((a, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.04 }}
                  className="rounded-lg border border-bg-border bg-bg-elevated p-2.5"
                >
                  <div className="flex items-start justify-between gap-2">
                    <span className="text-xs font-mono text-text-primary">{a.name}</span>
                    {a.stream !== "—" && <StreamBadge stream={a.stream} />}
                  </div>
                  <div className="flex items-center gap-2 mt-1">
                    <Tag>{a.type}</Tag>
                    <span className="text-[10px] text-text-muted">{a.note}</span>
                  </div>
                </motion.div>
              ))}
              {!expandAttrs && ds.attrs.length > 5 && (
                <button
                  onClick={() => setExpandAttrs(true)}
                  className="w-full text-xs text-sky-accent hover:text-sky-glow py-2 transition-colors font-mono"
                >
                  + {ds.attrs.length - 5} more attributes
                </button>
              )}
            </div>
          </Card>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}