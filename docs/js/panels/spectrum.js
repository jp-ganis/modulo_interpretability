// Panel 2: Fourier spectrum of the embedding. One bar per frequency k = 1..56;
// starts as an even spread of noise and sharpens into a few spikes.

import { changed } from "../state.js";
import { barPath, observeSize, tooltip } from "../common.js";

export function createSpectrumPanel(el, data, store) {
  const { meta, frames } = data;
  const nFreq = frames.spectrum[0].length;
  const ks = d3.range(1, nFreq + 1);
  const keySet = new Set(meta.key_freqs);
  const margin = { top: 16, right: 6, bottom: 22, left: 30 };
  const svg = d3.select(el).append("svg");
  const root = svg.append("g");
  let x, y, ih = 0;

  function render(W, H) {
    root.selectAll("*").remove();
    const iw = W - margin.left - margin.right;
    ih = H - margin.top - margin.bottom;
    if (iw < 40 || ih < 30) return;
    root.attr("transform", `translate(${margin.left},${margin.top})`);

    x = d3.scaleBand().domain(ks).range([0, iw]).paddingInner(0.25).paddingOuter(0.1);
    y = d3.scaleLinear().domain([0, meta.limits.spectrum]).range([ih, 0]).nice();

    root.append("g").attr("class", "gridline")
      .call(d3.axisLeft(y).ticks(Math.max(2, Math.floor(ih / 40))).tickSize(-iw).tickFormat(""));
    root.append("g").attr("class", "axis")
      .call(d3.axisLeft(y).ticks(Math.max(2, Math.floor(ih / 40))).tickSizeOuter(0));
    const xTicks = [1, 10, 20, 30, 40, 50].filter((k) => k <= nFreq);
    root.append("g").attr("class", "axis").attr("transform", `translate(0,${ih})`)
      .call(d3.axisBottom(x).tickValues(xTicks).tickSizeOuter(0));

    root.append("g").attr("class", "bars").selectAll("path").data(ks).join("path")
      .attr("class", (k) => (keySet.has(k) ? "bar key" : "bar"));
    root.append("g").selectAll("text").data(meta.key_freqs).join("text")
      .attr("class", "bar-label").attr("text-anchor", "middle")
      .attr("x", (k) => x(k) + x.bandwidth() / 2).text((k) => k);

    // Full-height invisible hit areas: easier to hover/click than thin bars
    root.append("g").selectAll("rect").data(ks).join("rect")
      .attr("class", "bar-hit").attr("x", (k) => x(k) - (x.step() - x.bandwidth()) / 2)
      .attr("width", x.step()).attr("y", 0).attr("height", ih)
      .style("cursor", (k) => (keySet.has(k) ? "pointer" : "default"))
      .on("pointermove", (event, k) => {
        const v = frames.spectrum[store.get().frame][k - 1];
        const note = keySet.has(k) ? " · key frequency" : "";
        tooltip.show(`<div class="tt-head">k = ${k}${note}</div>strength ${v.toFixed(2)}`, event);
      })
      .on("pointerleave", () => tooltip.hide())
      .on("click", (event, k) => {
        if (keySet.has(k)) store.set({ freqIdx: meta.key_freqs.indexOf(k) });
      });

    update();
  }

  function update() {
    if (!x) return;
    const { frame, freqIdx } = store.get();
    const spec = frames.spectrum[frame];
    const selected = meta.key_freqs[freqIdx];
    const w = x.bandwidth();
    root.selectAll(".bars path")
      .attr("d", (k) => barPath(x(k), y(spec[k - 1]), w, ih - y(spec[k - 1]), 2))
      .classed("selected", (k) => k === selected);
    root.selectAll(".bar-label").attr("y", (k) => y(spec[k - 1]) - 4);
  }

  observeSize(el, render);
  store.subscribe((state, prev) => {
    if (changed(state, prev, "frame", "freqIdx")) update();
  });
}
