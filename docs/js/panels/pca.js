// Panel 4: PCA of the embedding. "Final basis" projects every frame onto the
// final checkpoint's top-2 principal directions (a fixed camera); "per frame"
// uses each checkpoint's own top-2 directions, aligned to its neighbours so
// the view doesn't flip.

import { createScatter } from "./scatter.js";

export function createPcaPanel(el, data, store) {
  const { meta, frames } = data;
  createScatter(el, data, store, {
    points: (s) => (s.pcaBasis === "aligned" ? frames.pca_aligned : frames.pca)[s.frame],
    limit: (s) => (s.pcaBasis === "aligned" ? meta.limits.pca_aligned : meta.limits.pca),
    guideCircle: false,
    redrawOn: ["pcaBasis"],
    describe(a, s) {
      const pts = (s.pcaBasis === "aligned" ? frames.pca_aligned : frames.pca)[s.frame];
      return `PC1 ${pts[2 * a].toFixed(2)}, PC2 ${pts[2 * a + 1].toFixed(2)}`;
    },
  });
}
