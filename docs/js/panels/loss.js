// Panel 1: train/test loss (or accuracy), optionally with Nanda et al.'s
// progress measures, on a log step axis.

import { fmtLoss, fmtPct } from "../common.js";
import { createLineChart } from "./linechart.js";

export function createLossPanel(el, data, store) {
  const { curves } = data;
  const cp = curves.checkpoints;

  // y-range covering every loss-like series, fixed so toggling doesn't rescale
  const allLoss = [...curves.train_loss, ...curves.test_loss, ...cp.restricted_loss, ...cp.excluded_loss];
  const lossDomain = [d3.min(allLoss) * 0.5, d3.max(allLoss) * 1.6];

  createLineChart(el, data, store, {
    margin: { top: 18, right: 76, bottom: 22, left: 44 },
    yType: (s) => (s.lossMode === "acc" ? "linear" : "log"),
    redrawOn: ["lossMode", "showProgress"],
    phaseLabels: true,
    grokLine: true,
    directLabels: true,
    yDomain: (s) => (s.lossMode === "acc" ? [0, 1] : lossDomain),
    yFormat: (v) => {
      if (store.get().lossMode === "acc") return d3.format(".0%")(v);
      const e = Math.round(Math.log10(v));
      return e === 0 ? "1" : `1e${e}`;
    },
    valueFormat: (v) => (store.get().lossMode === "acc" ? fmtPct(v) : fmtLoss(v)),
    series(s) {
      if (s.lossMode === "acc") {
        return [
          { id: "train", label: "train", colour: "var(--train)", steps: curves.step, values: curves.train_acc },
          { id: "test", label: "test", colour: "var(--test)", steps: curves.step, values: curves.test_acc },
        ];
      }
      const out = [
        { id: "train", label: "train", colour: "var(--train)", steps: curves.step, values: curves.train_loss },
        { id: "test", label: "test", colour: "var(--test)", steps: curves.step, values: curves.test_loss },
      ];
      if (s.showProgress) {
        out.push(
          { id: "restricted", label: "restricted", colour: "var(--aqua)", steps: cp.step, values: cp.restricted_loss },
          { id: "excluded", label: "excluded", colour: "var(--ink-2)", steps: cp.step, values: cp.excluded_loss },
        );
      }
      return out;
    },
  });
}
