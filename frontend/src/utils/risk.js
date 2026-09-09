// Continuous risk-severity color scale: teal (low) -> amber (mid) -> deep red
// (high), interpolated so mid-risk reads as visually distinct from both
// extremes rather than a binary red/green split.
const STOPS = [
  { at: 0,    rgb: [15, 118, 110] },  // teal-700
  { at: 0.5,  rgb: [180, 83, 9] },    // amber-700
  { at: 1,    rgb: [153, 27, 27] },   // red-800
];

function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function getRiskColor(fraction) {
  const p = Math.min(1, Math.max(0, fraction));
  let lower = STOPS[0];
  let upper = STOPS[STOPS.length - 1];
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (p >= STOPS[i].at && p <= STOPS[i + 1].at) {
      lower = STOPS[i];
      upper = STOPS[i + 1];
      break;
    }
  }
  const span = upper.at - lower.at || 1;
  const t = (p - lower.at) / span;
  const rgb = lower.rgb.map((c, i) => Math.round(lerp(c, upper.rgb[i], t)));
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

export function getRiskLabel(fraction) {
  const pct = fraction * 100;
  if (pct < 25) return "Low";
  if (pct < 50) return "Elevated";
  if (pct < 75) return "High";
  return "Severe";
}
