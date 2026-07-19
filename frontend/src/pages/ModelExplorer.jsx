import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Layers, Brain, Zap, ChevronRight, Info, Code } from "lucide-react";
import { Card, SectionHeader, StreamBadge, Tag, Value, Button } from "@/components/ui";
import clsx from "clsx";

const STREAMS = [
  {
    id: "s1", label: "Stream 1", name: "ECG Waveform Encoder",
    color: "#38BDF8", bg: "bg-sky-accent/10", border: "border-sky-accent/20",
    icon: "〜", input: "12-lead ECG · 10s · 100Hz",
    output: "(B, 256)", model: "1D ResNet-18",
    layers: [
      { name: "Input Conv1D",     shape: "(B, 12, 1000) → (B, 64, 500)",  note: "kernel=16, stride=2" },
      { name: "ResBlock × 4",    shape: "(B, 64→128→256, …)",             note: "BatchNorm + ReLU" },
      { name: "Global Avg Pool", shape: "(B, 256, T) → (B, 256)",         note: "temporal compression" },
      { name: "Projection",      shape: "(B, 256)",                        note: "fusion-ready" },
    ],
    dataset: "PTB-XL raw waveforms (records100/500)",
    params: "~2.1M",
  },
  {
    id: "s2", label: "Stream 2", name: "Diagnostic Text Encoder",
    color: "#10B981", bg: "bg-safe/10", border: "border-safe/20",
    icon: "T", input: "Cardiologist diagnostic statements",
    output: "(B, 256)", model: "BioBERT fine-tuned",
    layers: [
      { name: "Tokenizer",        shape: "(B, seq≤128)",                   note: "WordPiece · SNOMED vocab" },
      { name: "BioBERT 12-layer", shape: "(B, 128, 768)",                  note: "biobert-base-cased-v1.2" },
      { name: "[CLS] extraction", shape: "(B, 768)",                        note: "sentence representation" },
      { name: "Linear proj.",     shape: "(B, 256)",                        note: "→ fusion dim" },
    ],
    dataset: "PTB-XL scp_statements + PTB-XL+ auto-diagnostics",
    params: "~110M (frozen backbone)",
  },
  {
    id: "s3", label: "Stream 3", name: "Structured Feature Encoder",
    color: "#F59E0B", bg: "bg-warning/10", border: "border-warning/20",
    icon: "≡", input: "ECG intervals, morphological features, demographics",
    output: "(B, 128)", model: "FT-Transformer",
    layers: [
      { name: "NumericalEmb",    shape: "(B, 32) → (B, 32, 128)",          note: "x_i · W_i + b_i per feature" },
      { name: "CatEmbedding",    shape: "(B, 6, 8) → (B, 6, 128)",        note: "sex, age_group, rhythm…" },
      { name: "FT-Transformer×3",shape: "(B, 38, 128)",                    note: "4 heads, pre-norm" },
      { name: "CLS token",        shape: "(B, 128)",                        note: "feature aggregation" },
    ],
    dataset: "PTB-XL+ features (glasgow, uni-leipzig, emreb)",
    params: "~1.8M",
  },
];

const FUSION = [
  { name: "Asymmetric Modality Trust (AMT)", desc: "3-stream concat → 2-layer MLP → softmax weights per step. Learns when to trust ECG over text.", key: true },
  { name: "Cross-Attention Layer × 3",       desc: "Each modality cross-attends to concatenated others, scaled by AMT weights. 8 heads, 256-dim.", key: false },
  { name: "Feed-Forward Block",               desc: "Standard pre-norm FFN with GELU activation and 0.1 dropout.", key: false },
  { name: "Aggregation",                      desc: "Concat 3 streams → Linear(768→256) → LayerNorm.", key: false },
  { name: "Multi-Horizon Head",               desc: "Attention-weighted temporal pooling → 3 binary classifiers (24h / 48h / 72h).", key: true },
];

export default function ModelExplorer() {
  const [active, setActive] = useState("s1");
  const [showCode, setShowCode] = useState(false);
  const stream = STREAMS.find(s => s.id === active);

  return (
    <div className="p-6 space-y-6">
      <SectionHeader
        eyebrow="Architecture"
        title="Three-Stream Multimodal Fusion"
        sub="Click a stream to inspect its layers, inputs, and dataset mapping."
      />

      {/* ── Stream selector tabs ─────────────────────────────────── */}
      <div className="grid grid-cols-3 gap-3">
        {STREAMS.map(s => (
          <motion.button
            key={s.id}
            whileHover={{ scale: 1.01 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => setActive(s.id)}
            className={clsx(
              "rounded-xl border p-4 text-left transition-all duration-150",
              active === s.id
                ? `${s.bg} ${s.border} shadow-lg`
                : "border-bg-border bg-bg-card hover:border-bg-border/80 hover:bg-bg-hover"
            )}
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-[10px] font-mono tracking-widest" style={{ color: s.color }}>
                {s.label}
              </span>
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center text-sm font-mono font-bold border"
                style={{ color: s.color, borderColor: `${s.color}30`, background: `${s.color}10` }}
              >
                {s.icon}
              </div>
            </div>
            <p className="font-display text-sm font-semibold text-text-primary mb-1">{s.name}</p>
            <p className="text-[10px] text-text-muted leading-relaxed">{s.input}</p>
            <div className="flex items-center gap-2 mt-2">
              <Tag>{s.model}</Tag>
              <Tag>{s.params}</Tag>
            </div>
          </motion.button>
        ))}
      </div>

      {/* ── Stream detail ─────────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
          className="grid grid-cols-1 lg:grid-cols-3 gap-4"
        >
          {/* Layer stack */}
          <Card className="lg:col-span-2">
            <div className="flex items-center justify-between mb-4">
              <SectionHeader eyebrow="Layer Stack" title={stream.name} />
              <Button
                variant="ghost" size="sm"
                onClick={() => setShowCode(c => !c)}
              >
                <Code size={12} /> {showCode ? "Hide" : "View"} shapes
              </Button>
            </div>
            <div className="space-y-2">
              {stream.layers.map((l, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.06 }}
                  className="flex items-start gap-3 rounded-lg border border-bg-border bg-bg-elevated p-3"
                >
                  <div
                    className="w-5 h-5 rounded flex items-center justify-center text-[9px] font-mono font-bold shrink-0 mt-0.5"
                    style={{ background: `${stream.color}20`, color: stream.color }}
                  >
                    {i + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-sm font-mono text-text-primary">{l.name}</span>
                      {showCode && (
                        <span className="text-[10px] font-mono text-text-muted bg-bg-base px-2 py-0.5 rounded border border-bg-border">
                          {l.shape}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-text-muted mt-0.5">{l.note}</p>
                  </div>
                  {i < stream.layers.length - 1 && (
                    <ChevronRight size={12} className="text-text-muted shrink-0 mt-1" />
                  )}
                </motion.div>
              ))}
            </div>
            {/* Output shape */}
            <div
              className="mt-3 rounded-lg border px-4 py-2.5 flex items-center justify-between"
              style={{ borderColor: `${stream.color}30`, background: `${stream.color}08` }}
            >
              <span className="text-xs text-text-muted font-mono">Output shape</span>
              <span className="text-sm font-mono font-semibold" style={{ color: stream.color }}>
                {stream.output}
              </span>
            </div>
          </Card>

          {/* Meta info */}
          <div className="space-y-4">
            <Card>
              <SectionHeader eyebrow="Dataset Source" title="Training Data" />
              <div className="bg-bg-elevated rounded-lg border border-bg-border px-3 py-2 mb-3">
                <p className="text-xs text-text-secondary font-mono leading-relaxed">{stream.dataset}</p>
              </div>
              <StreamBadge stream={stream.id.toUpperCase()} />
            </Card>

            <Card>
              <SectionHeader eyebrow="AMT Contribution" title="Trust at inference" />
              {STREAMS.map(s => (
                <div key={s.id} className="flex items-center gap-2 mb-2">
                  <div className="w-1.5 h-1.5 rounded-full" style={{ background: s.color }} />
                  <span className="text-xs text-text-secondary flex-1">{s.name.split(" ").slice(-1)[0]}</span>
                  <div className="text-xs font-mono" style={{ color: s.color }}>
                    {s.id === "s1" ? "~50%" : s.id === "s2" ? "~32%" : "~18%"}
                  </div>
                </div>
              ))}
              <p className="text-[10px] text-text-muted mt-2">Dynamic — varies per patient and time-step</p>
            </Card>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* ── Fusion block ─────────────────────────────────────────── */}
      <Card>
        <SectionHeader
          eyebrow="Cross-Modal Fusion"
          title="Asymmetric Modality Trust (AMT) — The Core Novelty"
          sub="Unlike models that average all streams equally, CardioSense learns dynamic per-step weights."
        />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {FUSION.map((f, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              className={clsx(
                "rounded-lg border p-3",
                f.key ? "border-sky-accent/30 bg-sky-accent/5" : "border-bg-border bg-bg-elevated"
              )}
            >
              <div className="flex items-start gap-2 mb-1.5">
                {f.key && <Zap size={12} className="text-sky-accent mt-0.5 shrink-0" />}
                {!f.key && <Layers size={12} className="text-text-muted mt-0.5 shrink-0" />}
                <p className="text-xs font-semibold font-mono text-text-primary leading-snug">{f.name}</p>
              </div>
              <p className="text-[11px] text-text-muted leading-relaxed pl-5">{f.desc}</p>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* ── Total params ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          ["ECG Encoder",   "~2.1M",  "#38BDF8"],
          ["BioBERT",       "~110M",  "#10B981"],
          ["Struct. Enc.",  "~1.8M",  "#F59E0B"],
          ["Fusion + Head", "~4.2M",  "#A78BFA"],
        ].map(([l, v, c]) => (
          <div key={l} className="rounded-xl border border-bg-border bg-bg-card px-4 py-3 text-center">
            <p className="text-[9px] font-mono text-text-muted tracking-widest mb-1">{l}</p>
            <p className="font-display text-lg font-semibold" style={{ color: c }}>{v}</p>
            <p className="text-[9px] text-text-muted">parameters</p>
          </div>
        ))}
      </div>
    </div>
  );
}