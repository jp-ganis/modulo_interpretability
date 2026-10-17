// The timeline: scrubber, play/pause, speed, readout, keyboard shortcuts, and
// tweenTo() for animated jumps (used by the story).
//
// The scrubber moves through *frames*, not raw steps. Frames are packed most
// densely where training changes fastest, so the grokking transition gets
// more of the slider than the long quiet stretches do.

import { changed } from "./state.js";
import { fmtLoss, fmtPct, fmtStep, reducedMotion } from "./common.js";
import { PHASE_NAMES } from "./data.js";

const BASE_FPS = 30; // frames per second at 1x: the 300-frame run plays in ~10 s

export function createControls(data, store) {
  const { meta, frames, curves, nFrames } = data;
  const slider = document.getElementById("scrubber");
  const playBtn = document.getElementById("play");
  const speedSel = document.getElementById("speed");
  const readout = document.getElementById("readout");
  slider.max = nFrames - 1;

  // Phase strip under the slider, positioned by frame index
  const strip = document.getElementById("phase-strip");
  const pct = (step) => (100 * data.nearestFrame(step)) / (nFrames - 1);
  const p = meta.phases;
  const bounds = [0, p.memorisation_end, p.cleanup_start, p.cleanup_end, meta.final_step];
  strip.innerHTML = bounds.slice(0, -1).map((b, i) =>
    `<span class="ph-${i}" style="left:${pct(b)}%;width:${pct(bounds[i + 1]) - pct(b)}%" title="${PHASE_NAMES[i]}"></span>`,
  ).join("") + `<span class="grok-tick" style="left:${pct(meta.grok_step)}%" title="Grokking (test accuracy ≥ 95%)"></span>`;

  // --- User input -------------------------------------------------------
  slider.addEventListener("input", () => store.set({ frame: +slider.value, playing: false, tweenId: 0 }));
  playBtn.addEventListener("click", () => {
    const s = store.get();
    const atEnd = s.frame >= nFrames - 1;
    store.set({ playing: !s.playing, frame: !s.playing && atEnd ? 0 : s.frame, tweenId: 0 });
  });
  speedSel.addEventListener("change", () => store.set({ speed: +speedSel.value }));

  document.addEventListener("keydown", (e) => {
    const tag = e.target.tagName;
    if (tag === "SELECT" || tag === "TEXTAREA" || (tag === "INPUT" && e.target.type !== "range")) return;
    if (e.key === " " && tag !== "BUTTON") {
      e.preventDefault();
      playBtn.click();
    } else if ((e.key === "ArrowRight" || e.key === "ArrowLeft") && e.target !== slider) {
      e.preventDefault();
      const f = store.get().frame + (e.key === "ArrowRight" ? 1 : -1);
      store.set({ frame: Math.max(0, Math.min(nFrames - 1, f)), playing: false, tweenId: 0 });
    }
  });

  // --- Playback loop ----------------------------------------------------
  let last = null, acc = 0;
  function tick(now) {
    const s = store.get();
    if (!s.playing) { last = null; return; }
    if (last != null) acc += ((now - last) / 1000) * BASE_FPS * s.speed;
    last = now;
    if (acc >= 1) {
      const next = Math.min(nFrames - 1, s.frame + Math.floor(acc));
      acc %= 1;
      store.set({ frame: next, playing: next < nFrames - 1 });
    }
    requestAnimationFrame(tick);
  }

  // --- Reflect state in the controls -------------------------------------
  function sync(s, prev) {
    if (!prev || changed(s, prev, "frame")) {
      slider.value = s.frame;
      const step = frames.step[s.frame];
      const i = data.nearestIndex(curves.step, step);
      readout.innerHTML =
        `<span>Step <b>${fmtStep(step)}</b></span>` +
        `<span>train loss <b>${fmtLoss(curves.train_loss[i])}</b></span>` +
        `<span>test loss <b>${fmtLoss(curves.test_loss[i])}</b></span>` +
        `<span>test acc <b>${fmtPct(curves.test_acc[i])}</b></span>` +
        `<span class="phase-name">${PHASE_NAMES[data.phaseOf(step)]}</span>`;
    }
    if (!prev || changed(s, prev, "playing")) {
      playBtn.classList.toggle("playing", s.playing);
      playBtn.setAttribute("aria-label", s.playing ? "Pause" : "Play");
      if (s.playing) { acc = 0; requestAnimationFrame(tick); }
    }
  }
  store.subscribe(sync);
  sync(store.get(), null);

  // --- Animated jump to a frame (cancelled by any user seek) --------------
  let tweenCounter = 0;
  function tweenTo(target) {
    const start = store.get().frame;
    const id = ++tweenCounter;
    const duration = reducedMotion() ? 0 : Math.min(2200, 400 + 6 * Math.abs(target - start));
    store.set({ tweenId: id, playing: false });
    const t0 = performance.now();
    function step(now) {
      if (store.get().tweenId !== id) return; // interrupted
      const t = duration ? Math.min(1, (now - t0) / duration) : 1;
      const f = Math.round(start + (target - start) * d3.easeCubicInOut(t));
      if (f !== store.get().frame) store.set({ frame: f });
      if (t < 1) requestAnimationFrame(step);
      else store.set({ tweenId: 0 });
    }
    requestAnimationFrame(step);
  }

  return { tweenTo };
}
