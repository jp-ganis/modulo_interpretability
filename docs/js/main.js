// Entry point: load the data, create the shared store, wire up the controls
// and the five linked panels, then the story.
//
// Why D3 + SVG: the charts need log scales, axes, and lines/bars that update
// every frame, which is D3's core job. The scatters have only 113 points, so
// SVG is easily fast enough, stays crisp at any size, and can be styled with
// CSS (dark mode is just a change of CSS variables). Canvas would only pay
// off with thousands of points. No build step: plain ES modules plus one
// vendored copy of D3.

import { createStore } from "./state.js";
import { loadData } from "./data.js";
import { createControls } from "./controls.js";
import { createLossPanel } from "./panels/loss.js";
import { createNormPanel } from "./panels/norm.js";
import { createSpectrumPanel } from "./panels/spectrum.js";
import { createRingPanel } from "./panels/ring.js";
import { createPcaPanel } from "./panels/pca.js";
import { initStory } from "./story.js";

initThemeToggle();

let data;
try {
  data = await loadData();
} catch (err) {
  console.error(err);
  document.querySelector(".panels").innerHTML =
    `<p class="load-error">Couldn't load the training data (${err.message}). If you opened this file directly,
     serve it instead: <code>python -m http.server -d docs</code>.</p>`;
  throw err;
}

const store = createStore({
  frame: 0,          // index into frames.step
  playing: false,
  speed: 1,
  tweenId: 0,        // non-zero while an animated jump is running
  freqIdx: 0,        // which key frequency the ring shows
  showLabels: false,
  connect: false,
  lossMode: "loss",  // "loss" | "acc"
  showProgress: false,
  pcaBasis: "fixed", // "fixed" | "aligned"
  hoverNum: null,    // number hovered in either scatter
});

fillText(data);
wireToggles(data, store);
const controls = createControls(data, store);
createLossPanel(document.getElementById("chart-loss"), data, store);
createNormPanel(document.getElementById("chart-norm"), data, store);
createSpectrumPanel(document.getElementById("chart-spectrum"), data, store);
createRingPanel(document.getElementById("chart-ring"), data, store);
createPcaPanel(document.getElementById("chart-pca"), data, store);
initStory(data, store, controls);

// --- Helpers ----------------------------------------------------------------

// Fill <span data-fill="..."> placeholders with numbers from this run, so the
// text stays correct if the model is retrained.
function fillText({ meta, curves }) {
  const fmt = d3.format(",");
  const words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
  const kf = meta.key_freqs;
  const values = {
    memorisation_end: fmt(meta.phases.memorisation_end),
    cleanup_start: fmt(meta.phases.cleanup_start),
    cleanup_end: fmt(meta.phases.cleanup_end),
    grok_step: fmt(meta.grok_step),
    final_step: fmt(meta.final_step),
    key_freqs: `k = ${kf.slice(0, -1).join(", ")} and ${kf.at(-1)}`,
    key_freqs_count: words[kf.length] ?? String(kf.length),
    test_loss_peak: Math.round(d3.max(curves.test_loss)),
    norm_start: Math.round(curves.weight_norm[0]),
    norm_peak: Math.round(d3.max(curves.weight_norm)),
    norm_final: Math.round(curves.weight_norm.at(-1)),
  };
  document.querySelectorAll("[data-fill]").forEach((el) => {
    el.textContent = values[el.dataset.fill] ?? "?";
  });
}

// Buttons and checkboxes <-> store (both directions, so the story can set them too)
function wireToggles({ meta }, store) {
  const freqBox = document.getElementById("freq-buttons");
  freqBox.innerHTML = meta.key_freqs.map((k, i) =>
    `<button type="button" data-freq-idx="${i}">k = ${k}</button>`).join("");
  freqBox.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) store.set({ freqIdx: +b.dataset.freqIdx });
  });
  document.querySelectorAll("[data-loss-mode]").forEach((b) =>
    b.addEventListener("click", () => store.set({ lossMode: b.dataset.lossMode })));
  document.querySelectorAll("[data-pca]").forEach((b) =>
    b.addEventListener("click", () => store.set({ pcaBasis: b.dataset.pca })));
  const checks = { "show-progress": "showProgress", "show-labels": "showLabels", "show-connect": "connect" };
  for (const [id, key] of Object.entries(checks)) {
    document.getElementById(id).addEventListener("change", (e) => store.set({ [key]: e.target.checked }));
  }

  // Colour legend for numbers: the same cyclic hue wheel the dots use
  const stops = meta.colours.filter((_, i) => i % 8 === 0).concat(meta.colours[0]);
  document.getElementById("hue-bar").style.background = `linear-gradient(90deg, ${stops.join(",")})`;

  function sync(s) {
    freqBox.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(+b.dataset.freqIdx === s.freqIdx)));
    document.getElementById("ring-k").textContent = `k = ${meta.key_freqs[s.freqIdx]}`;
    document.querySelectorAll("[data-loss-mode]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.lossMode === s.lossMode)));
    document.querySelectorAll("[data-pca]").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.pca === s.pcaBasis)));
    const prog = document.getElementById("show-progress");
    prog.checked = s.showProgress;
    prog.disabled = s.lossMode === "acc"; // progress measures are losses
    document.getElementById("show-labels").checked = s.showLabels;
    document.getElementById("show-connect").checked = s.connect;
  }
  store.subscribe(sync);
  sync(store.get());
}

// Light/dark toggle. The choice is remembered per browser (a convenience
// only, so storage failures are ignored).
function initThemeToggle() {
  const rootEl = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") rootEl.dataset.theme = saved;
  } catch {}
  document.getElementById("theme-toggle").addEventListener("click", () => {
    const current = rootEl.dataset.theme
      ?? (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    rootEl.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch {}
  });
}
