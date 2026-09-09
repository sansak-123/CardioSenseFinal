import React, { useMemo, useState } from "react";
import { format, formatDistanceToNow } from "date-fns";
import clsx from "clsx";
import {
  Activity, FileText, MessageSquare, HeartPulse, Info, ClipboardList,
} from "lucide-react";
import {
  ResponsiveContainer, AreaChart, Area, LineChart, Line, YAxis,
} from "recharts";
import ECGCanvas from "@/components/ECGCanvas";
import cases from "@/data/sample_cases.json";
import { getRiskColor, getRiskLabel } from "@/utils/risk";

const HORIZONS = [
  { key: "h24", label: "24 hours" },
  { key: "h48", label: "48 hours" },
  { key: "h72", label: "72 hours" },
];

const STREAM_META = {
  vitals:   { label: "ICU Vitals & Labs",        icon: Activity },
  notes:    { label: "Clinical Notes",            icon: FileText },
  symptoms: { label: "Patient-Reported Symptoms", icon: MessageSquare },
};

const STATUS_STYLE = {
  Stable:     "text-safe bg-safe/10 border-safe/30",
  Monitoring: "text-warning bg-warning/10 border-warning/30",
  Critical:   "text-critical bg-critical/10 border-critical/30",
};

const ECG_CLASS_LABEL = {
  NORM: "Normal",
  MI:   "Myocardial infarction",
  STTC: "ST/T-wave changes",
  CD:   "Conduction disturbance",
  HYP:  "Hypertrophy",
};

function pct(x) {
  return `${Math.round(x * 100)}%`;
}

/* ── Small building blocks ─────────────────────────────────────── */

function Panel({ eyebrow, title, sub, children, className }) {
  return (
    <div className={clsx("rounded-lg border border-bg-border bg-bg-card p-5", className)}>
      {(eyebrow || title) && (
        <div className="mb-4">
          {eyebrow && (
            <p className="text-[11px] font-medium text-text-muted tracking-wide uppercase mb-1">{eyebrow}</p>
          )}
          {title && <h2 className="text-base font-semibold text-text-primary">{title}</h2>}
          {sub && <p className="text-sm text-text-secondary mt-0.5">{sub}</p>}
        </div>
      )}
      {children}
    </div>
  );
}

function Sparkline({ data, color, uid }) {
  // Gradient id must be a valid SVG id — an rgb(...) color string contains
  // parens/commas/spaces, which silently breaks the url(#...) fill
  // reference, so a caller-supplied uid (not the raw color) names it.
  const gradientId = `spark-${uid}`;
  return (
    <ResponsiveContainer width="100%" height={36}>
      <AreaChart data={data.map((v, i) => ({ i, v }))} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.25} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.5}
              fill={`url(#${gradientId})`} dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

function CaseCard({ c, selected, onSelect }) {
  const color = getRiskColor(c.risk.h72);
  return (
    <button
      onClick={onSelect}
      className={clsx(
        "text-left rounded-lg border p-4 transition-colors bg-bg-card",
        selected ? "border-sky-accent ring-1 ring-sky-accent/30" : "border-bg-border hover:border-text-muted"
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm font-semibold text-text-primary">{c.id}</p>
          <p className="text-xs text-text-muted mt-0.5">
            {c.age}{c.sex} · Admitted {formatDistanceToNow(new Date(c.admittedAt), { addSuffix: true })}
          </p>
        </div>
        <span className={clsx("text-[10px] font-medium border rounded-full px-2 py-0.5", STATUS_STYLE[c.status])}>
          {c.status}
        </span>
      </div>
      <div className="flex items-center gap-2 mt-3">
        <span className="text-[11px] text-text-muted">72h risk</span>
        <span className="text-sm font-semibold" style={{ color }}>{pct(c.risk.h72)}</span>
      </div>
    </button>
  );
}

function RiskHorizonCard({ label, fraction, history }) {
  const color = getRiskColor(fraction);
  return (
    <div className="rounded-lg border border-bg-border bg-bg-card p-4">
      <p className="text-xs font-medium text-text-muted uppercase tracking-wide">{label}</p>
      <div className="flex items-baseline gap-2 mt-1.5">
        <span className="text-3xl font-semibold tabular-nums" style={{ color }}>{pct(fraction)}</span>
        <span className="text-xs font-medium" style={{ color }}>{getRiskLabel(fraction)}</span>
      </div>
      <div className="mt-2">
        <Sparkline data={history} color={color} uid={label.replace(/\s+/g, "-")} />
      </div>
      <p className="text-[10px] text-text-muted mt-1">Trend, last 7 hours</p>
    </div>
  );
}

function AttributionBar({ label, Icon, value, active }) {
  const width = Math.max(value * 100, active ? 2 : 0);
  return (
    <div className={clsx("py-2", !active && "opacity-40")}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <Icon size={14} className="text-text-secondary" />
          <span className="text-sm text-text-secondary">{label}</span>
        </div>
        <span className="text-sm font-medium text-text-primary tabular-nums">
          {active ? pct(value) : "excluded"}
        </span>
      </div>
      <div className="h-1.5 rounded-full bg-bg-hover overflow-hidden">
        <div
          className="h-full rounded-full bg-sky-accent transition-all duration-300"
          style={{ width: `${width}%` }}
        />
      </div>
    </div>
  );
}

function ModalityToggle({ label, Icon, active, onToggle, disabled }) {
  return (
    <div className="flex items-center justify-between py-2.5 border-b border-bg-border last:border-0">
      <div className="flex items-center gap-2">
        <Icon size={14} className={active ? "text-text-secondary" : "text-text-muted"} />
        <span className={clsx("text-sm", active ? "text-text-primary" : "text-text-muted")}>{label}</span>
      </div>
      <button
        onClick={onToggle}
        disabled={disabled}
        title={disabled ? "At least one modality must stay available" : undefined}
        className={clsx(
          "relative w-9 h-5 rounded-full transition-colors shrink-0",
          active ? "bg-sky-accent" : "bg-bg-border",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <span
          className={clsx(
            "absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform",
            active && "translate-x-4"
          )}
        />
      </button>
    </div>
  );
}

function VitalMini({ label, unit, series, dataKey, color, current }) {
  return (
    <div className="rounded-lg border border-bg-border p-3">
      <div className="flex items-baseline justify-between">
        <span className="text-[11px] font-medium text-text-muted uppercase tracking-wide">{label}</span>
        <span className="text-sm font-semibold text-text-primary tabular-nums">
          {current}<span className="text-[10px] font-normal text-text-muted ml-0.5">{unit}</span>
        </span>
      </div>
      <ResponsiveContainer width="100%" height={40}>
        <LineChart data={series} margin={{ top: 6, right: 0, bottom: 0, left: 0 }}>
          <YAxis hide domain={["dataMin - 4", "dataMax + 4"]} />
          <Line type="monotone" dataKey={dataKey} stroke={color} strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function EvidencePanel({ case_ }) {
  const last = case_.vitalsTrend[case_.vitalsTrend.length - 1];
  return (
    <Panel eyebrow="Supporting Evidence" title="What drove this prediction"
           sub="The actual inputs behind the numbers above, for sanity-checking against clinical judgment.">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
        <VitalMini label="Heart rate" unit="bpm" series={case_.vitalsTrend} dataKey="hr" color="#2C5282" current={last.hr} />
        <VitalMini label="Blood pressure (SBP)" unit="mmHg" series={case_.vitalsTrend} dataKey="sbp" color="#B45309" current={last.sbp} />
        <VitalMini label="SpO2" unit="%" series={case_.vitalsTrend} dataKey="spo2" color="#0F766E" current={last.spo2} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div>
          <p className="text-[11px] font-medium text-text-muted uppercase tracking-wide mb-1.5">Clinical note excerpt</p>
          <div className="rounded-lg bg-bg-elevated border border-bg-border p-3">
            <p className="text-sm text-text-secondary leading-relaxed">{case_.noteExcerpt}</p>
          </div>
        </div>
        <div>
          <p className="text-[11px] font-medium text-text-muted uppercase tracking-wide mb-1.5">Patient-reported symptoms</p>
          <div className="rounded-lg bg-bg-elevated border border-bg-border p-3">
            <p className="text-sm text-text-secondary leading-relaxed italic">&ldquo;{case_.symptomExcerpt}&rdquo;</p>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function ECGPanel({ ecg }) {
  const sorted = Object.entries(ecg.classes).sort((a, b) => b[1] - a[1]);
  const dominant = sorted[0][0];
  return (
    <div className="rounded-lg border border-sky-dim/30 bg-bg-elevated p-5">
      <div className="flex items-start justify-between mb-1">
        <div className="flex items-center gap-2">
          <HeartPulse size={16} className="text-sky-dim" />
          <h2 className="text-base font-semibold text-text-primary">ECG Risk Classifier</h2>
        </div>
        <span className="text-[10px] font-medium text-sky-dim bg-sky-accent/10 border border-sky-accent/25 rounded-full px-2.5 py-1">
          Separate model — not fused into the prediction above
        </span>
      </div>
      <p className="text-sm text-text-secondary mb-4">
        Trained independently on PTB-XL signal, engineered features, and demographics. Classifies diagnostic
        category from a 12-lead ECG; does not feed the 24/48/72h deterioration estimates.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <div>
          <p className="text-[11px] font-medium text-text-muted uppercase tracking-wide mb-2">Lead II (illustrative)</p>
          <div className="rounded-lg border border-bg-border bg-bg-card p-3">
            <ECGCanvas height={70} color="#334155" />
          </div>
        </div>
        <div>
          <p className="text-[11px] font-medium text-text-muted uppercase tracking-wide mb-2">Predicted diagnostic category</p>
          <div className="space-y-2">
            {sorted.map(([cls, score]) => (
              <div key={cls}>
                <div className="flex items-center justify-between mb-1">
                  <span className={clsx("text-sm", cls === dominant ? "text-text-primary font-medium" : "text-text-secondary")}>
                    {ECG_CLASS_LABEL[cls]} <span className="text-text-muted">({cls})</span>
                  </span>
                  <span className="text-sm font-medium text-text-primary tabular-nums">{pct(score)}</span>
                </div>
                <div className="h-1.5 rounded-full bg-bg-hover overflow-hidden">
                  <div
                    className={clsx("h-full rounded-full", cls === dominant ? "bg-sky-dim" : "bg-text-muted/50")}
                    style={{ width: `${score * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────────────────── */

export default function Dashboard() {
  const [selectedId, setSelectedId] = useState(cases[0].id);
  const [active, setActive] = useState({ vitals: true, notes: true, symptoms: true });

  const case_ = cases.find((c) => c.id === selectedId);
  const activeCount = Object.values(active).filter(Boolean).length;

  const selectCase = (id) => {
    setSelectedId(id);
    setActive({ vitals: true, notes: true, symptoms: true });
  };

  const toggle = (key) => {
    setActive((prev) => {
      if (prev[key] && activeCount === 1) return prev;
      return { ...prev, [key]: !prev[key] };
    });
  };

  const { fusedRisk, effectiveAttribution } = useMemo(() => {
    const activeKeys = Object.keys(active).filter((k) => active[k]);
    const weightSum = activeKeys.reduce((s, k) => s + case_.attribution[k], 0) || 1;

    const attribution = {};
    Object.keys(active).forEach((k) => {
      attribution[k] = active[k] ? case_.attribution[k] / weightSum : 0;
    });

    const risk = {};
    HORIZONS.forEach(({ key }) => {
      risk[key] = activeKeys.reduce(
        (sum, s) => sum + (case_.attribution[s] / weightSum) * case_.partialRisk[s][key],
        0
      );
    });

    return { fusedRisk: risk, effectiveAttribution: attribution };
  }, [active, case_]);

  return (
    <div className="p-6 space-y-6 max-w-6xl mx-auto">
      {/* Case selector */}
      <section>
        <p className="text-[11px] font-medium text-text-muted tracking-wide uppercase mb-2 flex items-center gap-1.5">
          <ClipboardList size={13} /> Sample cases
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {cases.map((c) => (
            <CaseCard key={c.id} c={c} selected={c.id === selectedId} onSelect={() => selectCase(c.id)} />
          ))}
        </div>
      </section>

      {/* Case header */}
      <div className="flex items-center justify-between border-b border-bg-border pb-4">
        <div>
          <p className="text-xs text-text-muted">
            {case_.id} · {case_.age}{case_.sex} · Admitted {format(new Date(case_.admittedAt), "MMM d, h:mm a")}
          </p>
          <h1 className="text-xl font-semibold text-text-primary mt-0.5">Deterioration Risk Assessment</h1>
        </div>
        <span className={clsx("text-xs font-medium border rounded-full px-3 py-1", STATUS_STYLE[case_.status])}>
          {case_.status}
        </span>
      </div>

      {/* Risk horizon panel */}
      <section>
        <p className="text-[11px] font-medium text-text-muted tracking-wide uppercase mb-2">
          Predicted risk of deterioration — trimodal model
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {HORIZONS.map((h) => (
            <RiskHorizonCard key={h.key} label={h.label} fraction={fusedRisk[h.key]} history={case_.riskHistory[h.key]} />
          ))}
        </div>
      </section>

      {/* Attribution + toggles */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel eyebrow="AMT Gate Output" title="Modality attribution"
               sub="How much each input stream contributed to the current prediction.">
          {Object.entries(STREAM_META).map(([key, meta]) => (
            <AttributionBar key={key} label={meta.label} Icon={meta.icon}
                             value={effectiveAttribution[key]} active={active[key]} />
          ))}
        </Panel>

        <Panel eyebrow="Robustness Check" title="Missing-modality stress test"
               sub="Simulate a stream being unavailable and watch the prediction adapt.">
          {Object.entries(STREAM_META).map(([key, meta]) => (
            <ModalityToggle key={key} label={meta.label} Icon={meta.icon} active={active[key]}
                             onToggle={() => toggle(key)} disabled={active[key] && activeCount === 1} />
          ))}
          {activeCount < 3 && (
            <p className="text-xs text-text-muted mt-3">
              Recomputed using only the {activeCount} remaining available stream{activeCount > 1 ? "s" : ""} —
              weights above are renormalized, not just zeroed.
            </p>
          )}
        </Panel>
      </div>

      {/* Supporting evidence */}
      <EvidencePanel case_={case_} />

      {/* ECG panel — separate model */}
      <ECGPanel ecg={case_.ecg} />
    </div>
  );
}
