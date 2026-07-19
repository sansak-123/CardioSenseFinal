import React, { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Activity, Heart, AlertTriangle, TrendingUp,
  Users, Clock, Zap, ArrowRight, Shield, ChevronRight
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid
} from "recharts";
import ECGCanvas from "@/components/ECGCanvas";
import { Card, StatCard, RiskBadge, SectionHeader, Tag, Button } from "@/components/ui";
import clsx from "clsx";

/* ── Mock data ─────────────────────────────────────────────────── */
const AUROC_DATA = [
  { epoch: 1,  ehr: 0.71, nlp: 0.68, fused: 0.72 },
  { epoch: 5,  ehr: 0.75, nlp: 0.72, fused: 0.78 },
  { epoch: 10, ehr: 0.78, nlp: 0.75, fused: 0.82 },
  { epoch: 15, ehr: 0.80, nlp: 0.77, fused: 0.84 },
  { epoch: 20, ehr: 0.81, nlp: 0.78, fused: 0.86 },
  { epoch: 25, ehr: 0.82, nlp: 0.79, fused: 0.874 },
  { epoch: 30, ehr: 0.82, nlp: 0.79, fused: 0.879 },
];

const RECENT_PATIENTS = [
  { id: "ECG-21401", age: 67, sex: "M", label: "MI",   risk24: 0.83, risk72: 0.91, status: "critical" },
  { id: "ECG-18834", age: 54, sex: "F", label: "STTC", risk24: 0.61, risk72: 0.74, status: "warning"  },
  { id: "ECG-09271", age: 42, sex: "M", label: "NORM", risk24: 0.09, risk72: 0.11, status: "safe"     },
  { id: "ECG-15502", age: 71, sex: "F", label: "HYP",  risk24: 0.48, risk72: 0.59, status: "warning"  },
  { id: "ECG-07839", age: 38, sex: "M", label: "CD",   risk24: 0.22, risk72: 0.28, status: "safe"     },
];

const STATUS_COLOR = {
  critical: "text-critical bg-critical/10 border-critical/30",
  warning:  "text-warning  bg-warning/10  border-warning/30",
  safe:     "text-safe     bg-safe/10     border-safe/30",
};

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-bg-elevated border border-bg-border rounded-lg p-3 text-xs font-mono">
      <p className="text-text-muted mb-1">Epoch {label}</p>
      {payload.map(p => (
        <p key={p.dataKey} style={{ color: p.color }}>{p.name}: {p.value.toFixed(3)}</p>
      ))}
    </div>
  );
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [tick, setTick] = useState(0);

  useEffect(() => {
    const id = setInterval(() => setTick(t => t + 1), 2000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="p-6 space-y-6">

      {/* ── Hero ECG banner ──────────────────────────────────────── */}
      <div className="relative rounded-2xl overflow-hidden border border-bg-border bg-bg-card">
        {/* Grid overlay */}
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(#38BDF8 1px, transparent 1px), linear-gradient(90deg, #38BDF8 1px, transparent 1px)",
            backgroundSize: "24px 24px",
          }}
        />
        <div className="relative z-10 px-8 py-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <p className="text-[10px] font-mono text-sky-accent tracking-[0.25em] mb-1">CARDIOSENSE · LIVE MONITOR</p>
              <h1 className="font-display text-3xl font-semibold text-text-primary leading-tight">
                Pre-Lab Cardiac<br />
                <span className="text-gradient">Deterioration Predictor</span>
              </h1>
              <p className="text-sm text-text-secondary mt-2 max-w-sm">
                Multimodal deep learning across ECG signals, morphological features,
                and diagnostic text. Predicts risk 24–72 h before lab confirmation.
              </p>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-2 justify-end mb-1">
                <div className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse" />
                <span className="text-[10px] font-mono text-safe">STREAMING</span>
              </div>
              <p className="font-display text-4xl font-bold text-gradient">0.879</p>
              <p className="text-[10px] text-text-muted font-mono mt-0.5">AUROC @ 72h</p>
            </div>
          </div>
          <ECGCanvas height={80} color="#38BDF8" className="w-full opacity-90" />
        </div>
      </div>

      {/* ── Stat row ─────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Training Records"   value="21,837" unit="ECGs"    sub="PTB-XL dataset"      icon={Activity}  accent="sky"      />
        <StatCard label="Model AUROC"        value="87.9"   unit="%"       sub="72h horizon"         icon={TrendingUp} accent="safe"     />
        <StatCard label="Critical Alerts"    value="3"      unit="active"  trend={12} sub="last 24h" icon={AlertTriangle} accent="critical" />
        <StatCard label="Patients Monitored" value="1,204"  unit="total"   sub="this session"        icon={Users}     accent="sky"      />
      </div>

      {/* ── Middle row ───────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* AUROC chart */}
        <Card className="lg:col-span-2">
          <SectionHeader
            eyebrow="Training Progress"
            title="AUROC by Modality Stream"
            sub="Fusion consistently outperforms single-stream baselines"
          />
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={AUROC_DATA} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                {[["fused","#38BDF8"],["nlp","#10B981"],["ehr","#F59E0B"]].map(([k,c]) => (
                  <linearGradient key={k} id={`g-${k}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={c} stopOpacity={0.15} />
                    <stop offset="95%" stopColor={c} stopOpacity={0} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E3A5F50" />
              <XAxis dataKey="epoch" tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} />
              <YAxis domain={[0.65, 0.92]} tick={{ fontSize: 10, fill: "#475569", fontFamily: "JetBrains Mono" }} tickFormatter={v => v.toFixed(2)} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="fused" name="Fused"     stroke="#38BDF8" fill="url(#g-fused)" strokeWidth={2} dot={false} />
              <Area type="monotone" dataKey="nlp"   name="NLP Text"  stroke="#10B981" fill="url(#g-nlp)"   strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
              <Area type="monotone" dataKey="ehr"   name="ECG Only"  stroke="#F59E0B" fill="url(#g-ehr)"   strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
            </AreaChart>
          </ResponsiveContainer>
          <div className="flex gap-4 mt-2">
            {[["#38BDF8","Fused (CardioSense)"],["#10B981","NLP Text"],["#F59E0B","ECG Only"]].map(([c,l]) => (
              <div key={l} className="flex items-center gap-1.5">
                <div className="w-3 h-0.5" style={{ background: c }} />
                <span className="text-[10px] text-text-muted font-mono">{l}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* Horizon risk cards */}
        <Card>
          <SectionHeader eyebrow="Prediction" title="Risk Horizons" />
          <div className="space-y-2.5">
            <RiskBadge score={0.61} label="24h Deterioration" />
            <RiskBadge score={0.74} label="48h Deterioration" />
            <RiskBadge score={0.83} label="72h Deterioration" />
          </div>
          <div className="mt-4 pt-4 border-t border-bg-border">
            <p className="text-[10px] font-mono text-text-muted mb-2">MODALITY TRUST WEIGHTS</p>
            {[["ECG Signal","52%","#38BDF8"],["Clinical Text","32%","#10B981"],["Structured","16%","#F59E0B"]].map(([l,v,c]) => (
              <div key={l} className="flex items-center gap-2 mb-1.5">
                <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: c }} />
                <span className="text-xs text-text-secondary flex-1">{l}</span>
                <span className="text-xs font-mono" style={{ color: c }}>{v}</span>
              </div>
            ))}
          </div>
          <Button onClick={() => navigate("/patient")} variant="outline" size="sm" className="w-full mt-3">
            Full Analysis <ArrowRight size={12} />
          </Button>
        </Card>
      </div>

      {/* ── Recent patients ───────────────────────────────────────── */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <SectionHeader eyebrow="Queue" title="Recent ECG Records" />
          <Button variant="ghost" size="sm" onClick={() => navigate("/patient")}>
            View all <ChevronRight size={12} />
          </Button>
        </div>
        <div className="space-y-0">
          {RECENT_PATIENTS.map((p, i) => (
            <motion.div
              key={p.id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
              onClick={() => navigate("/patient")}
              className="flex items-center justify-between py-3 border-b border-bg-border last:border-0 hover:bg-bg-hover rounded px-2 -mx-2 cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-bg-elevated border border-bg-border flex items-center justify-center">
                  <Heart size={13} className="text-text-muted" />
                </div>
                <div>
                  <p className="text-xs font-mono text-text-primary">{p.id}</p>
                  <p className="text-[10px] text-text-muted">{p.age}y {p.sex} · {p.label}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-right hidden sm:block">
                  <p className="text-[10px] text-text-muted font-mono">24h / 72h</p>
                  <p className="text-xs font-mono text-text-secondary">
                    {Math.round(p.risk24 * 100)}% / {Math.round(p.risk72 * 100)}%
                  </p>
                </div>
                <span className={clsx(
                  "text-[9px] font-mono border rounded px-2 py-0.5 tracking-wide uppercase",
                  STATUS_COLOR[p.status]
                )}>
                  {p.status}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </Card>

      {/* ── Bottom callout ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {[
          { icon: Zap,    label: "Pre-Lab Window",      val: "24–72h",   sub: "Before abnormal labs" },
          { icon: Shield, label: "Modality Trust",      val: "AMT Gate",  sub: "Dynamic per time-step" },
          { icon: Clock,  label: "Inference Speed",     val: "<200ms",   sub: "Per patient" },
        ].map(({ icon: Icon, label, val, sub }) => (
          <div key={label} className="flex items-center gap-3 rounded-xl border border-bg-border bg-bg-card px-4 py-3">
            <div className="w-8 h-8 rounded-lg bg-sky-accent/10 border border-sky-accent/20 flex items-center justify-center shrink-0">
              <Icon size={14} className="text-sky-accent" />
            </div>
            <div>
              <p className="text-[10px] text-text-muted font-mono">{label}</p>
              <p className="text-sm font-display font-semibold text-text-primary">{val}</p>
              <p className="text-[10px] text-text-muted">{sub}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}