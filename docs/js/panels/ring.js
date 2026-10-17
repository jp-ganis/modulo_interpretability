// Panel 3: the ring for one key frequency k. Every number's embedding is
// projected onto the final model's cos/sin plane for k (a fixed basis for
// all frames), scaled so a finished ring has radius 1 with number a at
// angle 2*pi*k*a/p.

import { createScatter } from "./scatter.js";

export function createRingPanel(el, data, store) {
  const { meta, frames } = data;
  createScatter(el, data, store, {
    points: (s) => frames.rings[s.frame][s.freqIdx],
    limit: (s) => meta.limits.ring[s.freqIdx], // per frequency: k = 8 is much noisier early on
    guideCircle: true,
    redrawOn: ["freqIdx"],
    describe(a, s) {
      const k = meta.key_freqs[s.freqIdx];
      // Where a sits on the finished ring, counted in 1/p-ths of a turn
      return `ring position ${(k * a) % meta.p} of ${meta.p}<br><span class="muted">(${k} × ${a} mod ${meta.p})</span>`;
    },
  });
}
