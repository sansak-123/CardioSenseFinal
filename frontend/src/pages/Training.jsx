import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Play, Pause, RotateCcw, Terminal, TrendingUp,
  CheckCircle2, Clock, Cpu, Download, AlertCircle
} from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, BarChart, Bar, Cell
} from "recharts";
import { Card, SectionHeader, StatCard, Tag, Button, Value } from "@/components/ui";
import clsx from "clsx";

/* ── Simulated training data ───────────────────────────────────── */
const FULL_HISTORY = Array.from({ length: 30 }, (_, i) => ({
  epoch: i + 1,
  train_loss: +(1.2 * Math.exp(-i * 0.12) + 0.08 + Math.random() * 0.02).toFixed(4),
  val_loss:   +(1.3 * Math.exp(-i * 0.10) + 0.11 + Math.random() * 0.03).toFixed(4),
  auroc_24h:  +(0.70 + (i / 30) * 0.15 + Math.random() * 0.01).toFixed(4),
  auroc_48h:  +(0.71 + (i / 30) * 0.16 + Math.random() * 0.01).toFixed(4),
  auroc_72h:  +(0.72 + (i / 30) * 0.16 + Math.random() * 0.01).toFixed(4),
}));

const ABLATION = [
  { name: "SOFA Score (baseline)",    auroc: 0.74, color: "#475569" },
  { name: "NEWS2 Score",              auroc: 0.77, color: "#64748B" },
  { name: "EHR-only LSTM",           auroc: 0.81, color: "#94A3B8" },
  { name: "S1 only (ECG signal)",    auroc: 0.83, color: "#F59E0B" },
  { name: "S1 + S2 (ECG + Text)",   auroc: 0.85, color: "#10B981" },
  { name: "CardioSense (full)",      auroc: 0.879, color: "#38BDF8" },
];

const LOG_LINES = [
  "[INFO]  Loading PTB-XL dataset from ./data/ptb-xl/",
  "[INFO]  21,837 records loaded. Train: 17,441 | Val: 2,183 | Test: 2,183",
  "[INFO]  Class weights: NORM=0.57 MI=0.99 STTC=1.03 CD=1.10 HYP=1.96",
  "[INFO]  Initialising 1D-ResNet ECG encoder (2.1M params)",
  "[INFO]  Loading BioBERT: dmis-lab/biobert-base-cased-v1.2 (110M params)",
  "[INFO]  FT-Transformer structured encoder (1.8M params)",
  "[INFO]  Fusion + AMT gate + multi-horizon head (4.2M params)",
  "[INFO]  Total trainable: ~118.1M parameters",
  "[TRAIN] Epoch 1/50 — loss: 1.2341, val_loss: 1.3105, AUROC@72h: 0.7201",
  "[TRAIN] Epoch 5/50 — loss: 0.8412, val_loss: 0.9234, AUROC@72h: 0.7834",
  "[TRAIN] Epoch 10/50 — loss: 0.5231, val_loss: 0.6102, AUROC@72h: 0.8213",
  "[TRAIN] Epoch 15/50 — loss: 0.3847, val_loss: 0.4592, AUROC@72h: 0.8401",
  "[TRAIN] Epoch 20/50 — loss: 0.2934, val_loss: 0.3712, AUROC@72h: 0.8601",
  "[TRAIN] Epoch 25/50 — loss: 0.2471, val_loss: 0.3203, AUROC@72h: 0.8740",
  "[TRAIN] Epoch 30/50 — loss: 0.2318, val_loss: 0.3089, AUROC@72h: 0.8790",
  "[INFO]  Early stopping patience: 10/10",
  "[INFO]  Best checkpoint: epoch_30_auroc0.879.pt",
  "[EVAL]  Test AUROC  24h: 0.857  48h: 0.868  72h: 0.879",
  "[EVAL]  Test AUPRC  24h: 0.641  48h: 0.667  72h: 0.691",
  "[EVAL]  Brier score: 0.112",
  "[DONE]  Training complete.",
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-bg-border rounded-lg p-3 text-xs font-mono space-y-1">
      <p className="text-text-muted">Epoch {label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>{p.name}: {p.value}</p>
      ))}
    </div>
  );
};

export default function Training() {
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(30);
  const [logIdx, setLogIdx] = useState(LOG_LINES.length);
  const logRef = useRef(null);

  const displayed = FULL_HISTORY.slice(0, epoch);
  const current = displayed[displayed.length - 1];

  useEffect(() => {
    if (!running) return;
    if (epoch >= 30) { setRunning(false); return; }
    const id = setTimeout(() => {
      setEpoch(e => e + 1);
      setLogIdx(i => Math.min(i + 1, LOG_LINES.length));
    }, 350);
    return () => clearTimeout(id);
  }, [running, epoch]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logIdx]);

  const reset = () => { setEpoch(1); setLogIdx(8); setRunning(false); };

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-start justify-between">
        <SectionHeader
          eyebrow="Training"
          title="Model Training & Evaluation"
          sub="50 epochs · AdamW · Cosine LR · Weighted BCE + severity aux loss"
        />
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={reset}>
            <RotateCcw size={12} /> Reset
          </Button>
          <Button
            variant={running ? "critical" : "primary"}
            size="sm"
            onClick={() => { if (epoch >= 30) reset(); setRunning(r => !r); }}
          >
            {running ? <><Pause size={12} /> Pause</> : <><Play size={12} /> {epoch >= 30 ? "Replay" : "Resume"}</>}
          </Button>
        </div>
      </div>

      {/* ── Live stat row ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Current Epoch" value={epoch} unit="/ 50"      icon={TrendingUp} accent="sky"      />
        <StatCard label="AUROC @ 72h"  value={current ? (current.auroc_72h * 100).toFixed(1) : "—"} unit="%"  icon={CheckCircle2} accent="safe" />
        <StatCard label="Val Loss"     value={current?.val_loss ?? "—"}               icon={AlertCircle}  accent="warning"  />
        <StatCard label="Status"       value={epoch >= 30 ? "Done" : running ? "Running" : "Paused"} icon={Cpu} accent={epoch >= 30 ? "safe" : "sky"} />
      </div>

      {/* ── Charts ────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Loss curves */}
        <Card>
          <SectionHeader eyebrow="Loss" title="Train vs Validation Loss" />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={displayed} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F40" />
              <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
              <YAxis tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="train_loss" name="Train" stroke="#38BDF8" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="val_loss"   name="Val"   stroke="#E11D48" strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        {/* AUROC curves */}
        <Card>
          <SectionHeader eyebrow="Evaluation" title="AUROC by Horizon" />
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={displayed} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F40" />
              <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
              <YAxis domain={[0.68, 0.92]} tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip />} />
              <Line type="monotone" dataKey="auroc_24h" name="24h" stroke="#F59E0B" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="auroc_48h" name="48h" stroke="#10B981" strokeWidth={1.5} dot={false} />
              <Line type="monotone" dataKey="auroc_72h" name="72h" stroke="#38BDF8" strokeWidth={2}   dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>

      {/* ── Ablation + terminal ───────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Ablation study */}
        <Card>
          <SectionHeader
            eyebrow="Ablation"
            title="Baseline Comparison"
            sub="CardioSense vs single-stream and clinical score baselines"
          />
          <div className="space-y-2">
            {ABLATION.map((a, i) => (
              <div key={i} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-text-secondary font-mono">{a.name}</span>
                  <span className="text-xs font-mono" style={{ color: a.color }}>{a.auroc.toFixed(3)}</span>
                </div>
                <div className="h-1.5 rounded-full bg-bg-border overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${((a.auroc - 0.7) / 0.2) * 100}%` }}
                    transition={{ duration: 0.6, delay: i * 0.07 }}
                    className="h-full rounded-full"
                    style={{ background: a.color }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-bg-border grid grid-cols-3 gap-2">
            {[["AUROC","0.879"],["AUPRC","0.691"],["Brier","0.112"]].map(([k,v]) => (
              <div key={k} className="text-center">
                <p className="text-[9px] font-mono text-text-muted">{k}</p>
                <p className="text-sm font-display font-semibold text-text-primary">{v}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Training terminal */}
        <Card className="flex flex-col">
          <div className="flex items-center gap-2 mb-3">
            <Terminal size={13} className="text-sky-accent" />
            <SectionHeader eyebrow="Console" title="Training Log" />
            {running && <div className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse ml-auto" />}
          </div>
          <div
            ref={logRef}
            className="flex-1 bg-bg-base rounded-lg border border-bg-border p-3 overflow-y-auto font-mono text-[10px] leading-relaxed space-y-0.5"
            style={{ minHeight: 220, maxHeight: 280 }}
          >
            {LOG_LINES.slice(0, logIdx).map((line, i) => (
              <motion.p
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className={clsx(
                  line.startsWith("[INFO]")  && "text-text-muted",
                  line.startsWith("[TRAIN]") && "text-sky-accent",
                  line.startsWith("[EVAL]")  && "text-safe",
                  line.startsWith("[DONE]")  && "text-warning font-semibold",
                  !line.match(/^\[/) && "text-text-muted"
                )}
              >
                {line}
              </motion.p>
            ))}
            {running && (
              <p className="text-sky-accent animate-pulse">█</p>
            )}
          </div>
          <div className="flex gap-2 mt-3">
            <Button variant="ghost" size="sm" className="text-[10px]">
              <Download size={11} /> Export logs
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px]">
              <Download size={11} /> Save checkpoint
            </Button>
          </div>
        </Card>
      </div>

      {/* ── Hyperparameter table ─────────────────────────────────── */}
      <Card>
        <SectionHeader eyebrow="Config" title="Hyperparameters" sub="From utils/config.py" />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-0 divide-x-0">
          {[
            ["Epochs", "50"],     ["Batch size", "64"],      ["Eff. batch", "256"],
            ["LR", "1e-4"],       ["Weight decay", "1e-4"],  ["Warmup", "5 epochs"],
            ["Grad clip", "1.0"], ["Label smooth", "0.1"],   ["Scheduler", "cosine"],
            ["ECG sample", "100 Hz"], ["Seq. length", "10s"], ["Leads", "12"],
          ].map(([k, v]) => (
            <div key={k} className="flex items-center justify-between py-2.5 px-3 border-b border-bg-border">
              <span className="text-[11px] text-text-muted">{k}</span>
              <span className="text-[11px] font-mono text-text-primary">{v}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}