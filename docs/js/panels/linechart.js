// A reusable "metric vs training step" chart, used by the loss and weight
// norm panels. Log-scale step axis with phase bands, a marker for the
// current frame, a hover crosshair with tooltip, and click-to-seek.

import { changed } from "../state.js";
import { fmtStep, observeSize, tooltip, ttRow } from "../common.js";

const PHASE_SHORT = ["memorisation", "circuit formation", "cleanup"];

/**
 * opts:
 *   series(state)  -> [{ id, label, colour, steps: number[], values: number[] }]
 *   yType          "log" | "linear", or a function of state returning one
 *   yDomain(state) -> [lo, hi]
 *   yFormat        tick formatter
 *   valueFormat    tooltip formatter
 *   redrawOn       state keys that change the series or axes
 *   margin         { top, right, bottom, left }
 */
export function createLineChart(el, data, store, opts) {
  const { meta } = data;
  const margin = opts.margin;
  const svg = d3.select(el).append("svg");
  const root = svg.append("g");
  let x, y, W = 0, H = 0, series = [];

  // Step 0 can't sit on a log axis, so everything is plotted at step + 1.
  const X = (step) => x(step + 1);

  function render() {
    const state = store.get();
    series = opts.series(state);
    root.selectAll("*").remove();
    const iw = W - margin.left - margin.right;
    const ih = H - margin.top - margin.bottom;
    if (iw < 40 || ih < 30) return;
    root.attr("transform", `translate(${margin.left},${margin.top})`);

    const yType = typeof opts.yType === "function" ? opts.yType(state) : opts.yType;
    x = d3.scaleLog().domain([1, meta.final_step + 1]).range([0, iw]);
    y = (yType === "log" ? d3.scaleLog() : d3.scaleLinear()).domain(opts.yDomain(state)).range([ih, 0]);
    if (yType === "linear") y.nice();

    // Phase bands: shade memorisation and cleanup, leave circuit formation clear.
    const p = meta.phases;
    const edges = [0, p.memorisation_end, p.cleanup_start, p.cleanup_end];
    [0, 2].forEach((i) => {
      root.append("rect").attr("class", "phase-band")
        .attr("x", X(edges[i])).attr("width", X(edges[i + 1]) - X(edges[i]))
        .attr("y", 0).attr("height", ih);
    });
    edges.slice(1).forEach((e) => {
      root.append("line").attr("class", "phase-edge").attr("x1", X(e)).attr("x2", X(e)).attr("y1", 0).attr("y2", ih);
    });
    if (opts.phaseLabels && iw > 420) {
      PHASE_SHORT.forEach((name, i) => {
        // Centre labels in their band, except cleanup: it's narrow on a log
        // axis, so its label starts at the band edge and runs right.
        const lo = Math.max(X(edges[i]), 0), hi = X(edges[i + 1]);
        const cleanup = i === 2;
        root.append("text").attr("class", "phase-label")
          .attr("x", cleanup ? lo + 4 : (lo + hi) / 2).attr("y", -5)
          .attr("text-anchor", cleanup ? "start" : "middle").text(name);
      });
    }

    // Axes and grid
    const xTicks = [1, 10, 100, 1000, 10000, 100000].filter((t) => t <= meta.final_step + 1);
    root.append("g").attr("class", "gridline").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickValues(xTicks).tickSize(-ih).tickFormat(""));
    root.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickValues(xTicks).tickFormat(d3.format("~s")).tickSizeOuter(0));
    const yAxis = d3.axisLeft(y).tickFormat(opts.yFormat).tickSizeOuter(0);
    if (yType === "log") {
      // Powers of ten only, thinned to every 2nd/3rd decade when space is tight
      const [lo, hi] = y.domain().map(Math.log10);
      const decades = d3.range(Math.ceil(lo), Math.floor(hi) + 1);
      const every = Math.ceil(decades.length / Math.max(2, Math.floor(ih / 22)));
      yAxis.tickValues(decades.filter((e) => e % every === 0).map((e) => 10 ** e));
    } else {
      yAxis.ticks(Math.max(2, Math.floor(ih / 40)));
    }
    root.append("g").attr("class", "axis").call(yAxis);

    // Grokking moment
    if (opts.grokLine) {
      root.append("line").attr("class", "grok-line")
        .attr("x1", X(meta.grok_step)).attr("x2", X(meta.grok_step)).attr("y1", 0).attr("y2", ih);
      root.append("text").attr("class", "grok-label")
        .attr("x", X(meta.grok_step) - 4).attr("y", ih - 6).attr("text-anchor", "end").text("grok");
    }

    // Lines
    const line = d3.line().x((d) => X(d[0])).y((d) => y(Math.max(d[1], y.domain()[0])));
    for (const s of series) {
      root.append("path").attr("class", `line ${s.id}`)
        .attr("d", line(d3.zip(s.steps, s.values)));
    }

    // Direct labels at the right end, nudged apart so they don't overlap
    if (opts.directLabels) {
      const labels = series.map((s) => ({ s, y: y(Math.max(s.values.at(-1), y.domain()[0])) }))
        .sort((a, b) => a.y - b.y);
      for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 12);
      for (const l of labels) {
        root.append("text").attr("class", "line-label")
          .attr("x", X(meta.final_step) + 6).attr("y", l.y).attr("dy", "0.35em").text(l.s.label);
      }
    }

    // Current-frame marker and dots, then the hover layer on top
    root.append("line").attr("class", "now-line").attr("y1", 0).attr("y2", ih);
    root.selectAll(".dot").data(series).join("circle").attr("class", (s) => `dot ${s.id}`).attr("r", 4);
    root.append("line").attr("class", "hover-line").attr("y1", 0).attr("y2", ih).style("display", "none");
    root.append("rect").attr("class", "hit").attr("width", iw).attr("height", ih)
      .attr("fill", "transparent").style("cursor", "pointer")
      .on("pointermove", hover).on("pointerleave", leave).on("click", seek);

    update();
  }

  // Value of each series at a step (nearest logged point)
  const valueAt = (s, step) => s.values[data.nearestIndex(s.steps, step)];

  function update() {
    if (!x) return;
    const step = data.frames.step[store.get().frame];
    root.select(".now-line").attr("x1", X(step)).attr("x2", X(step));
    root.selectAll(".dot")
      .attr("cx", X(step))
      .attr("cy", (s) => y(Math.max(valueAt(s, step), y.domain()[0])));
  }

  const stepFromEvent = (event) => Math.max(0, Math.round(x.invert(d3.pointer(event)[0]) - 1));

  function hover(event) {
    const step = Math.min(stepFromEvent(event), meta.final_step);
    root.select(".hover-line").style("display", null).attr("x1", X(step)).attr("x2", X(step));
    const rows = series.map((s) => ttRow(s.label, s.colour, opts.valueFormat(valueAt(s, step)))).join("");
    tooltip.show(`<div class="tt-head">Step ${fmtStep(step)}</div>${rows}`, event);
  }
  function leave() {
    root.select(".hover-line").style("display", "none");
    tooltip.hide();
  }
  function seek(event) {
    store.set({ frame: data.nearestFrame(stepFromEvent(event)), playing: false, tweenId: 0 });
  }

  observeSize(el, (w, h) => { W = w; H = h; render(); });
  store.subscribe((state, prev) => {
    if (opts.redrawOn && changed(state, prev, ...opts.redrawOn)) render();
    else if (changed(state, prev, "frame")) update();
  });
}
