import React from "react";
import clsx from "clsx";
import { motion } from "framer-motion";

/* ── Card ──────────────────────────────────────────────────────── */
export function Card({ children, className, glow = false, onClick }) {
  return (
    <motion.div
      whileHover={onClick ? { scale: 1.005 } : undefined}
      onClick={onClick}
      className={clsx(
        "rounded-xl border border-bg-border bg-bg-card p-5 transition-colors",
        glow && "border-glow",
        onClick && "cursor-pointer hover:border-sky-accent/30",
        className
      )}
    >
      {children}
    </motion.div>
  );
}

/* ── Stat card ─────────────────────────────────────────────────── */
export function StatCard({ label, value, unit, sub, trend, icon: Icon, accent = "sky" }) {
  const colors = {
    sky:      "text-sky-accent bg-sky-accent/10 border-sky-accent/20",
    critical: "text-critical bg-critical/10 border-critical/20",
    safe:     "text-safe bg-safe/10 border-safe/20",
    warning:  "text-warning bg-warning/10 border-warning/20",
  };
  return (
    <Card>
      <div className="flex items-start justify-between mb-3">
        <p className="text-xs text-text-muted font-mono tracking-widest uppercase">{label}</p>
        {Icon && (
          <div className={clsx("w-7 h-7 rounded-lg border flex items-center justify-center", colors[accent])}>
            <Icon size={13} />
          </div>
        )}
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="font-display text-2xl font-semibold text-text-primary">{value}</span>
        {unit && <span className="text-xs text-text-muted font-mono">{unit}</span>}
      </div>
      {(sub || trend) && (
        <div className="flex items-center gap-2 mt-1.5">
          {trend !== undefined && (
            <span className={clsx("text-xs font-mono", trend > 0 ? "text-critical" : "text-safe")}>
              {trend > 0 ? "▲" : "▼"} {Math.abs(trend)}%
            </span>
          )}
          {sub && <span className="text-[11px] text-text-muted">{sub}</span>}
        </div>
      )}
    </Card>
  );
}

/* ── Risk badge ────────────────────────────────────────────────── */
export function RiskBadge({ score, label }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "critical" : pct >= 40 ? "warning" : "safe";
  const colorMap = {
    critical: { text: "text-critical", bg: "bg-critical/10", border: "border-critical/30", bar: "bg-critical" },
    warning:  { text: "text-warning",  bg: "bg-warning/10",  border: "border-warning/30",  bar: "bg-warning" },
    safe:     { text: "text-safe",     bg: "bg-safe/10",     border: "border-safe/30",     bar: "bg-safe" },
  }[color];

  return (
    <div className={clsx("rounded-lg border px-3 py-2", colorMap.bg, colorMap.border)}>
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-[10px] font-mono text-text-muted tracking-widest">{label}</span>
        <span className={clsx("text-sm font-display font-semibold", colorMap.text)}>{pct}%</span>
      </div>
      <div className="h-1 rounded-full bg-bg-border overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={clsx("h-full rounded-full", colorMap.bar)}
        />
      </div>
    </div>
  );
}

/* ── Section header ────────────────────────────────────────────── */
export function SectionHeader({ eyebrow, title, sub }) {
  return (
    <div className="mb-6">
      {eyebrow && (
        <p className="text-[10px] font-mono text-sky-accent tracking-[0.2em] uppercase mb-1">{eyebrow}</p>
      )}
      <h2 className="font-display text-xl font-semibold text-text-primary">{title}</h2>
      {sub && <p className="text-sm text-text-secondary mt-1">{sub}</p>}
    </div>
  );
}

/* ── Stream badge ──────────────────────────────────────────────── */
export function StreamBadge({ stream }) {
  const map = {
    "S1": { label: "Stream 1 · ECG Signal",       cls: "text-sky-accent  bg-sky-accent/10  border-sky-accent/20"  },
    "S2": { label: "Stream 2 · Clinical NLP",      cls: "text-safe        bg-safe/10        border-safe/20"        },
    "S3": { label: "Stream 3 · Structured",        cls: "text-warning     bg-warning/10     border-warning/20"     },
  };
  const { label, cls } = map[stream] || { label: stream, cls: "text-text-muted bg-bg-elevated border-bg-border" };
  return (
    <span className={clsx("text-[9px] font-mono border rounded px-1.5 py-0.5 tracking-wide", cls)}>
      {label}
    </span>
  );
}

/* ── Tag ───────────────────────────────────────────────────────── */
export function Tag({ children, color = "default" }) {
  const colors = {
    default:  "bg-bg-elevated  text-text-secondary border-bg-border",
    sky:      "bg-sky-accent/10  text-sky-accent  border-sky-accent/20",
    critical: "bg-critical/10 text-critical border-critical/20",
    safe:     "bg-safe/10     text-safe     border-safe/20",
    warning:  "bg-warning/10  text-warning  border-warning/20",
  };
  return (
    <span className={clsx("text-[10px] font-mono border rounded px-2 py-0.5", colors[color])}>
      {children}
    </span>
  );
}

/* ── Divider ───────────────────────────────────────────────────── */
export function Divider({ className }) {
  return <div className={clsx("h-px bg-bg-border", className)} />;
}

/* ── Inline value ──────────────────────────────────────────────── */
export function Value({ label, value, mono = false }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-bg-border last:border-0">
      <span className="text-xs text-text-muted">{label}</span>
      <span className={clsx("text-xs text-text-primary", mono && "font-mono")}>{value}</span>
    </div>
  );
}

/* ── Button ────────────────────────────────────────────────────── */
export function Button({ children, onClick, variant = "primary", size = "md", className, disabled }) {
  const variants = {
    primary:  "bg-sky-accent text-bg-base hover:bg-sky-glow font-medium",
    outline:  "border border-sky-accent/40 text-sky-accent hover:bg-sky-accent/10",
    ghost:    "text-text-secondary hover:text-text-primary hover:bg-bg-hover",
    critical: "bg-critical/10 border border-critical/30 text-critical hover:bg-critical/20",
  };
  const sizes = { sm: "px-3 py-1.5 text-xs", md: "px-4 py-2 text-sm", lg: "px-6 py-3 text-sm" };
  return (
    <motion.button
      whileHover={{ scale: 1.01 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "rounded-lg transition-colors duration-150 font-body inline-flex items-center gap-2 disabled:opacity-40 disabled:cursor-not-allowed",
        variants[variant], sizes[size], className
      )}
    >
      {children}
    </motion.button>
  );
}