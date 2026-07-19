import React, { useState } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  Activity, LayoutDashboard, User, Brain,
  Database, TrendingUp, ChevronLeft, ChevronRight,
  Wifi, AlertCircle
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";

const NAV = [
  { to: "/",          icon: LayoutDashboard, label: "Dashboard",        sub: "Overview" },
  { to: "/patient",   icon: User,            label: "Patient Analysis",  sub: "Risk scoring" },
  { to: "/model",     icon: Brain,           label: "Model Explorer",    sub: "Architecture" },
  { to: "/datasets",  icon: Database,        label: "Datasets",          sub: "PTB-XL & more" },
  { to: "/training",  icon: TrendingUp,      label: "Training",          sub: "Metrics & logs" },
];

export default function Layout() {
  const [collapsed, setCollapsed] = useState(false);
  const location = useLocation();

  return (
    <div className="flex h-screen bg-bg-base overflow-hidden">
      {/* ── Sidebar ────────────────────────────────────────────── */}
      <motion.aside
        animate={{ width: collapsed ? 64 : 224 }}
        transition={{ duration: 0.25, ease: [0.4, 0, 0.2, 1] }}
        className="relative flex flex-col border-r border-bg-border bg-bg-card shrink-0 z-20"
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-bg-border overflow-hidden">
          <div className="w-8 h-8 rounded-lg bg-sky-accent/10 border border-sky-accent/30 flex items-center justify-center shrink-0">
            <Activity size={16} className="text-sky-accent" />
          </div>
          <AnimatePresence>
            {!collapsed && (
              <motion.div
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.15 }}
                className="overflow-hidden"
              >
                <p className="font-display font-600 text-sm text-text-primary tracking-wide leading-none">CardioSense</p>
                <p className="text-[10px] text-text-muted mt-0.5 font-mono">v1.0 · PTB-XL</p>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Status dot */}
        <div className={clsx(
          "flex items-center gap-2 px-4 py-2.5 mx-3 mt-3 rounded-lg bg-safe/5 border border-safe/20",
          collapsed && "justify-center px-0 mx-2"
        )}>
          <div className="w-1.5 h-1.5 rounded-full bg-safe animate-pulse-slow shrink-0" />
          {!collapsed && <span className="text-[10px] font-mono text-safe tracking-widest">MODEL READY</span>}
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 space-y-0.5 px-2 overflow-hidden">
          {NAV.map(({ to, icon: Icon, label, sub }) => (
            <NavLink key={to} to={to} end={to === "/"}>
              {({ isActive }) => (
                <motion.div
                  whileHover={{ x: collapsed ? 0 : 2 }}
                  className={clsx(
                    "flex items-center gap-3 rounded-lg px-2.5 py-2.5 transition-colors duration-150 group",
                    isActive
                      ? "bg-sky-accent/10 border border-sky-accent/20"
                      : "hover:bg-bg-hover border border-transparent",
                    collapsed && "justify-center px-0"
                  )}
                >
                  <Icon
                    size={16}
                    className={clsx(
                      "shrink-0 transition-colors",
                      isActive ? "text-sky-accent" : "text-text-muted group-hover:text-text-secondary"
                    )}
                  />
                  <AnimatePresence>
                    {!collapsed && (
                      <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.1 }}
                        className="overflow-hidden"
                      >
                        <p className={clsx(
                          "text-sm font-medium leading-none",
                          isActive ? "text-text-primary" : "text-text-secondary group-hover:text-text-primary"
                        )}>{label}</p>
                        <p className="text-[10px] text-text-muted mt-0.5">{sub}</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Collapse toggle */}
        <div className="px-2 pb-4">
          <button
            onClick={() => setCollapsed(c => !c)}
            className="w-full flex items-center justify-center gap-2 py-2 rounded-lg border border-bg-border hover:border-bg-border/80 hover:bg-bg-hover transition-colors text-text-muted hover:text-text-secondary"
          >
            {collapsed ? <ChevronRight size={14} /> : <ChevronLeft size={14} />}
            {!collapsed && <span className="text-xs">Collapse</span>}
          </button>
        </div>
      </motion.aside>

      {/* ── Main content ───────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-6 py-3 border-b border-bg-border bg-bg-card/60 backdrop-blur-sm shrink-0">
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-muted font-mono">
              {location.pathname === "/" ? "DASHBOARD" :
               location.pathname.slice(1).toUpperCase().replace("-", " ")}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-1.5 text-text-muted">
              <Wifi size={12} />
              <span className="text-[10px] font-mono">PTB-XL · 21,837 ECGs</span>
            </div>
            <div className="w-px h-4 bg-bg-border" />
            <div className="flex items-center gap-1.5">
              <div className="w-6 h-6 rounded-full bg-sky-accent/10 border border-sky-accent/20 flex items-center justify-center">
                <span className="text-[9px] font-mono text-sky-accent">AI</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
              className="h-full"
            >
              <Outlet />
            </motion.div>
          </AnimatePresence>
        </main>
      </div>
    </div>
  );
}