// Scroll-driven story. Each <article class="step"> says which moment of
// training to show (data-at) and how to set up the panels (data-freq,
// data-progress, data-connect, data-labels).
//
// Wide screens: when a step crosses the middle of the viewport, the
// dashboard (pinned beside the story) animates to that moment.
// Narrow screens: the dashboard isn't pinned, so each step gets a
// "Show me" button that scrolls up to the panels and animates.

export function initStory(data, store, controls) {
  const steps = [...document.querySelectorAll(".step")];
  const wide = window.matchMedia("(min-width: 1200px)");
  let active = null;

  function apply(el) {
    const d = el.dataset;
    const k = Number(d.freq);
    const freqIdx = data.meta.key_freqs.includes(k) ? data.meta.key_freqs.indexOf(k) : store.get().freqIdx;
    store.set({
      freqIdx,
      lossMode: "loss",
      showProgress: d.progress === "true",
      connect: d.connect === "true",
      showLabels: d.labels === "true",
    });
    controls.tweenTo(data.nearestFrame(data.resolveStep(d.at)));
    active = el;
    steps.forEach((s) => s.classList.toggle("active", s === el));
  }

  // A step counts as "reached" when it crosses a thin band in the middle of the viewport
  const observer = new IntersectionObserver((entries) => {
    if (!wide.matches) return;
    for (const e of entries) if (e.isIntersecting && e.target !== active) apply(e.target);
  }, { rootMargin: "-45% 0px -50% 0px" });
  steps.forEach((s) => observer.observe(s));

  steps.forEach((s) => s.querySelector(".show-me")?.addEventListener("click", () => {
    document.getElementById("dash").scrollIntoView({ behavior: "smooth", block: "start" });
    apply(s);
  }));

  // On wide screens the first step is visible on load: mark it without animating
  if (wide.matches) steps[0].classList.add("active");
}
