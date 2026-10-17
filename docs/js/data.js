// Loads the JSON exported by `python -m grokking.export` and adds a few
// lookup helpers the panels share.

export const PHASE_NAMES = ["Memorisation", "Circuit formation", "Cleanup", "Generalised"];

export async function loadData(base = "data/") {
  const get = (name) =>
    fetch(base + name).then((r) => {
      if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
      return r.json();
    });
  const [meta, curves, frames] = await Promise.all([get("meta.json"), get("curves.json"), get("frames.json")]);

  const frameSteps = frames.step;
  const nFrames = frameSteps.length;
  const bisectStep = d3.bisector((d) => d).left;

  // Index of the frame whose step is closest to `step`.
  function nearestFrame(step) {
    const i = bisectStep(frameSteps, step);
    if (i <= 0) return 0;
    if (i >= nFrames) return nFrames - 1;
    return step - frameSteps[i - 1] <= frameSteps[i] - step ? i - 1 : i;
  }

  // Index into a sorted step array closest to `step` (used for curves).
  function nearestIndex(steps, step) {
    const i = bisectStep(steps, step);
    if (i <= 0) return 0;
    if (i >= steps.length) return steps.length - 1;
    return step - steps[i - 1] <= steps[i] - step ? i - 1 : i;
  }

  // 0 memorisation, 1 circuit formation, 2 cleanup, 3 after cleanup
  function phaseOf(step) {
    const p = meta.phases;
    if (step < p.memorisation_end) return 0;
    if (step < p.cleanup_start) return 1;
    if (step < p.cleanup_end) return 2;
    return 3;
  }

  // Named moments used by the story (data attributes like data-at="grok_step").
  function resolveStep(at) {
    const p = meta.phases;
    const named = {
      final: meta.final_step,
      grok_step: meta.grok_step,
      memorisation_end: p.memorisation_end,
      cleanup_start: p.cleanup_start,
      cleanup_end: p.cleanup_end,
      // late in circuit formation: the progress measures have clearly moved,
      // but test loss hasn't yet
      circuit_late: Math.round(p.memorisation_end + 0.85 * (p.cleanup_start - p.memorisation_end)),
    };
    return at in named ? named[at] : Number(at);
  }

  return { meta, curves, frames, nFrames, nearestFrame, nearestIndex, phaseOf, resolveStep };
}
