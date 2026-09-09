import React, { useState } from "react";
import { Outlet, NavLink, useLocation } from "react-router-dom";
import {
  HeartPulse, LayoutDashboard, UserPlus,
  ChevronLeft, ChevronRight,
  Info
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";

const NAV = [
  { to: "/",             icon: LayoutDashboard, label: "Dashboard",     sub: "Deterioration risk" },
  { to: "/new-patient",  icon: UserPlus,        label: "New Patient",   sub: "Live prediction" },
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
            <HeartPulse size={16} className="text-sky-accent" />
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
                <p className="text-sm font-semibold text-text-primary tracking-wide leading-none">CardioSentinel</p>
                <p className="text-[10px] text-text-muted mt-0.5">Research demo</p>
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
          {!collapsed && <span className="text-[10px] text-safe tracking-wide">Model ready</span>}
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
        <header className="flex items-center justify-between px-6 py-3 border-b border-bg-border bg-bg-card shrink-0">
          <span className="text-xs font-medium text-text-secondary">
            {location.pathname === "/" ? "Dashboard" :
             location.pathname.slice(1).replace("-", " ").replace(/^\w/, (c) => c.toUpperCase())}
          </span>
        </header>

        {/* Research-demo disclaimer — visible on every page */}
        <div className="flex items-center gap-2 px-6 py-2 border-b border-bg-border bg-bg-elevated shrink-0">
          <Info size={13} className="text-text-muted shrink-0" />
          <p className="text-[11px] text-text-muted">
            Research/capstone demo — not a certified clinical device. Patient cases shown are synthetic sample data.
          </p>
        </div>

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