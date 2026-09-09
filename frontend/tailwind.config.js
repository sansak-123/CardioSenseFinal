/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // Restrained clinical palette — light neutrals, one muted accent
        // reserved for interactive/active state, risk severity carried by
        // riskScale below rather than by decorative color elsewhere.
        bg: {
          base:    "#F3F5F7",
          card:    "#FFFFFF",
          elevated:"#F8FAFC",
          border:  "#E2E8F0",
          hover:   "#EEF2F6",
        },
        sky: {
          accent:  "#2C5282",
          dim:     "#1E3A5F",
          glow:    "#4A6FA5",
        },
        critical: "#B91C1C",
        warning:  "#B45309",
        safe:     "#0F766E",
        text: {
          primary:   "#0F172A",
          secondary: "#475569",
          muted:     "#94A3B8",
          accent:    "#2C5282",
        },
        // Continuous risk-severity scale (used only for risk indicators —
        // horizon cards, sparklines, attribution when tied to risk).
        risk: {
          low:  "#0F766E",
          mid:  "#B45309",
          high: "#991B1B",
        },
      },
      fontFamily: {
        display: ["'Space Grotesk'", "sans-serif"],
        body:    ["'Inter'", "sans-serif"],
        mono:    ["'JetBrains Mono'", "monospace"],
      },
      animation: {
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "glow":       "glow 2s ease-in-out infinite alternate",
        "scan":       "scan 2s linear infinite",
      },
      keyframes: {
        glow: {
          "0%":   { boxShadow: "0 0 5px #38BDF820" },
          "100%": { boxShadow: "0 0 20px #38BDF840, 0 0 40px #38BDF820" },
        },
        scan: {
          "0%":   { transform: "translateY(0%)" },
          "100%": { transform: "translateY(100%)" },
        }
      },
      backdropBlur: { xs: "2px" },
    }
  },
  plugins: [],
}