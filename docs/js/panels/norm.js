// Panel 5: total weight norm, sqrt(sum of every parameter squared).

import { createLineChart } from "./linechart.js";

export function createNormPanel(el, data, store) {
  const { curves } = data;
  const [lo, hi] = d3.extent(curves.weight_norm);

  createLineChart(el, data, store, {
    margin: { top: 10, right: 12, bottom: 22, left: 34 },
    yType: "linear",
    redrawOn: null,
    phaseLabels: false,
    grokLine: true,
    directLabels: false,
    yDomain: () => [lo - (hi - lo) * 0.08, hi + (hi - lo) * 0.08],
    yFormat: d3.format("~f"),
    valueFormat: d3.format(".2f"),
    series: () => [
      { id: "norm", label: "weight norm", colour: "var(--ink-2)", steps: curves.step, values: curves.weight_norm },
    ],
  });
}
