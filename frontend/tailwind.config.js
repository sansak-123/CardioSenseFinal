/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: {
          base:    "#060E1F",
          card:    "#0A1628",
          elevated:"#0F1E35",
          border:  "#1E3A5F",
          hover:   "#162240",
        },
        sky: {
          accent:  "#38BDF8",
          dim:     "#0EA5E9",
          glow:    "#7DD3FC",
        },
        critical: "#E11D48",
        warning:  "#F59E0B",
        safe:     "#10B981",
        text: {
          primary:   "#F8FAFC",
          secondary: "#94A3B8",
          muted:     "#475569",
          accent:    "#38BDF8",
        }
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