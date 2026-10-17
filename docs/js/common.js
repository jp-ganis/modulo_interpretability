// Shared helpers for the panels: number formatting, the tooltip, resizing.

export const fmtStep = d3.format(",");
export const fmtPct = d3.format(".1%");

// Losses span 1e-8 .. 30, so switch to scientific notation for small values.
export function fmtLoss(v) {
  if (v == null || !isFinite(v)) return "–";
  return v >= 0.01 ? d3.format(".3~r")(v) : d3.format(".2~e")(v);
}

// --- Tooltip: one shared floating element --------------------------------

const tipEl = () => document.getElementById("tooltip");

export const tooltip = {
  show(html, event) {
    const el = tipEl();
    el.innerHTML = html;
    el.hidden = false;
    const pad = 14;
    const { innerWidth: vw, innerHeight: vh } = window;
    const r = el.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + r.width > vw - 8) x = event.clientX - r.width - pad; // flip left near the edge
    if (y + r.height > vh - 8) y = event.clientY - r.height - pad;
    el.style.left = `${Math.max(8, x)}px`;
    el.style.top = `${Math.max(8, y)}px`;
  },
  hide() {
    tipEl().hidden = true;
  },
};

// Tooltip row with a coloured key, e.g. row("test", "var(--test)", "0.25")
export function ttRow(label, colour, value) {
  return `<div class="tt-row"><i style="background:${colour}"></i><span>${label}</span><span>${value}</span></div>`;
}

// --- Resizing --------------------------------------------------------------

// Calls cb(width, height) whenever the element's size changes (batched per frame).
export function observeSize(el, cb) {
  let pending = false;
  const ro = new ResizeObserver(() => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const { width, height } = el.getBoundingClientRect();
      if (width > 0 && height > 0) cb(width, height);
    });
  });
  ro.observe(el);
}

// Rounded-top bar path anchored to the baseline (square bottom corners).
export function barPath(x, y, w, h, r) {
  r = Math.min(r, w / 2, h);
  if (h <= 0) return "";
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}

export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
