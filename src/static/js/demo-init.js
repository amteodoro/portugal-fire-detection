// Demo-mode bootstrap. Runs on every page that injects __DEMO_STATE__.
// Reads window.__DEMO_STATE__ (server-rendered), shows the banner,
// adds body.demo-mode for CSS-driven hiding/disabling, and exposes
// helpers used by app.js / dataset.html / inference.html.
(function () {
  const state = window.__DEMO_STATE__ || { demo: false };
  window.__DEMO__ = !!state.demo;
  window.__SNAPSHOT_RANGE__ = state.snapshot_range || null;
  window.__SNAPSHOT_COUNTS__ = state.counts || {};

  if (!state.demo) return;

  // Mark body so CSS can scope demo-only styling.
  document.body.classList.add('demo-mode');

  // Populate banner.
  const banner = document.getElementById('demoBanner');
  if (banner) {
    const r = state.snapshot_range || {};
    const range = r.start && r.end ? `${r.start} → ${r.end}` : 'frozen snapshot';
    banner.innerHTML =
      `<strong>DEMO MODE</strong> — read-only snapshot of Portugal fires (${range}). ` +
      `Live data fetching, AOI subscriptions, training, and cache management are disabled.`;
    banner.style.display = 'block';
  }

  // Disable any element marked with [data-demo-disabled] (CSS handles
  // .demo-disabled visuals). For form controls, also set disabled=true
  // so they cannot be activated by keyboard.
  document.querySelectorAll('[data-demo-disabled]').forEach(el => {
    el.classList.add('demo-disabled');
    if ('disabled' in el) el.disabled = true;
    el.setAttribute('aria-disabled', 'true');
    if (!el.title) el.title = 'Disabled in demo mode';
  });

  // Replace the FIRMS time-period helper text with the actual snapshot
  // range so the user knows exactly which window the visible data covers.
  const r = state.snapshot_range || {};
  const info = document.getElementById('fireDaysInfo');
  if (info && r.start && r.end) {
    info.textContent = `Snapshot window: ${r.start} → ${r.end} (locked in demo mode)`;
  }
  // Pre-fill historical timeline date inputs to the snapshot window so
  // a user clicking "Search History" (when re-enabled outside demo)
  // sees a sensible default.
  const histStart = document.getElementById('histStart');
  const histEnd = document.getElementById('histEnd');
  if (histStart && r.start) histStart.value = r.start;
  if (histEnd && r.end) histEnd.value = r.end;
})();
