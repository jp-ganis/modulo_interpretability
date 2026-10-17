// A square scatter of the p numbers, shared by the ring and PCA panels.
// Points are coloured by number on a cyclic hue wheel (112 sits next to 0).
// Hovering a point highlights the same number in every scatter (state.hoverNum).

import { changed } from "../state.js";
import { observeSize, tooltip } from "../common.js";

/**
 * opts:
 *   points(state) -> flat [x0, y0, x1, y1, ...] for numbers 0..p-1
 *   limit(state)  -> half-width of the fixed square axis range
 *   guideCircle   draw a dashed radius-1 circle (where a finished ring sits)
 *   describe(a, state) -> extra tooltip text for number a
 *   redrawOn      state keys that change the axis range
 */
export function createScatter(el, data, store, opts) {
  const { meta } = data;
  const p = meta.p;
  const nums = d3.range(p);
  const svg = d3.select(el).append("svg");
  const root = svg.append("g");
  let s = null; // data -> pixel scale (same for x and y; y is flipped)

  function render(W, H) {
    root.selectAll("*").remove();
    const size = Math.min(W, H);
    const pad = 16;
    const L = opts.limit(store.get());
    s = d3.scaleLinear().domain([-L, L]).range([-(size / 2 - pad), size / 2 - pad]);
    root.attr("transform", `translate(${W / 2},${H / 2})`);

    const r = s.range()[1];
    // Clip to the plot square: the axis range deliberately ignores the rare
    // far-out points of the noisy early frames.
    const clipId = `clip-${el.id}`;
    root.append("clipPath").attr("id", clipId).append("rect")
      .attr("x", -size / 2).attr("y", -size / 2).attr("width", size).attr("height", size);
    root.attr("clip-path", `url(#${clipId})`);
    root.append("line").attr("class", "axis-cross").attr("x1", -r).attr("x2", r);
    root.append("line").attr("class", "axis-cross").attr("y1", -r).attr("y2", r);
    if (opts.guideCircle) root.append("circle").attr("class", "guide").attr("r", s(1) - s(0));

    root.append("path").attr("class", "connect");
    root.append("g").attr("class", "pts").selectAll("circle").data(nums).join("circle")
      .attr("class", "pt").attr("r", 4).attr("fill", (a) => meta.colours[a]);
    root.append("g").attr("class", "labels").selectAll("text").data(nums).join("text")
      .attr("class", "pt-label").text((a) => a);

    // "Connect a -> a+1" highlight: the first three steps, 0 -> 1 -> 2 -> 3
    root.append("path").attr("class", "connect-head").attr("marker-end", null);
    root.append("g").attr("class", "head-labels").selectAll("text").data([0, 1, 2, 3]).join("text")
      .attr("class", "connect-head-label").attr("text-anchor", "middle").attr("dominant-baseline", "middle")
      .text((a) => a);

    svg.on("pointermove", hover).on("pointerleave", leave);
    update();
  }

  // Pixel position of number a (y flipped so "up" is positive)
  const pos = (pts, a) => [s(pts[2 * a]), -s(pts[2 * a + 1])];

  // Push a label away from the centre so it sits just outside its dot
  function labelPos(pts, a, gap) {
    const [px, py] = pos(pts, a);
    const d = Math.hypot(px, py) || 1;
    return [px + (px / d) * gap, py + (py / d) * gap];
  }

  function update() {
    if (!s) return;
    const state = store.get();
    const pts = opts.points(state);

    root.selectAll(".pts circle")
      .attr("cx", (a) => pos(pts, a)[0]).attr("cy", (a) => pos(pts, a)[1])
      .classed("hovered", (a) => a === state.hoverNum)
      .attr("r", (a) => (a === state.hoverNum ? 6 : 4));
    if (state.hoverNum != null) root.selectAll(".pts circle").filter((a) => a === state.hoverNum).raise();

    root.select(".labels").style("display", state.showLabels ? null : "none");
    if (state.showLabels) {
      root.selectAll(".labels text")
        .attr("x", (a) => labelPos(pts, a, 9)[0]).attr("y", (a) => labelPos(pts, a, 9)[1]);
    }

    const showConnect = state.connect;
    const line = d3.line();
    root.select(".connect").style("display", showConnect ? null : "none")
      .attr("d", showConnect ? line([...nums, 0].map((a) => pos(pts, a))) : null);
    root.select(".connect-head").style("display", showConnect ? null : "none")
      .attr("d", showConnect ? line([0, 1, 2, 3].map((a) => pos(pts, a))) : null);
    root.select(".head-labels").style("display", showConnect && !state.showLabels ? null : "none");
    if (showConnect) {
      root.selectAll(".head-labels text")
        .attr("x", (a) => labelPos(pts, a, 12)[0]).attr("y", (a) => labelPos(pts, a, 12)[1]);
    }
  }

  // Nearest point within 16px of the pointer
  function hover(event) {
    if (!s) return;
    const [mx, my] = d3.pointer(event, root.node());
    const pts = opts.points(store.get());
    let best = null, bestD = 16;
    for (const a of nums) {
      const [px, py] = pos(pts, a);
      const d = Math.hypot(px - mx, py - my);
      if (d < bestD) { best = a; bestD = d; }
    }
    if (best !== store.get().hoverNum) store.set({ hoverNum: best });
    if (best == null) tooltip.hide();
    else tooltip.show(`<div class="tt-head">number ${best}</div>${opts.describe(best, store.get())}`, event);
  }
  function leave() {
    tooltip.hide();
    if (store.get().hoverNum != null) store.set({ hoverNum: null });
  }

  let size = [0, 0];
  observeSize(el, (w, h) => { size = [w, h]; render(w, h); });
  store.subscribe((state, prev) => {
    if (opts.redrawOn && changed(state, prev, ...opts.redrawOn)) render(...size);
    else if (changed(state, prev, "frame", "freqIdx", "hoverNum", "showLabels", "connect", "pcaBasis")) update();
  });
}
