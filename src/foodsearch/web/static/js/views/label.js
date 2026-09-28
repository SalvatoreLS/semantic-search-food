import { api } from "../api.js";
import { esc, icon } from "../dom.js";
import { RUBRIC, price } from "../format.js";

const GRADE_NAMES = { 0: "Irrelevant", 1: "Partial", 2: "Good", 3: "Perfect" };
const state = { queue: null, idx: 0, marks: {}, done: 0, total: 0, rubricOpen: true, error: "", saving: false };

function progress() {
  const pct = state.total ? ((state.done / state.total) * 100).toFixed(1) : 0;
  return `
<div class="fs-progress">
<span class="fs-eyebrow">Progress</span>
<div class="fs-bar" role="progressbar" aria-label="Pairs labeled" aria-valuemin="0" aria-valuemax="${state.total}" aria-valuenow="${state.done}"><span style="width: ${pct}%"></span></div>
<span class="fs-mono" style="font-size: 15px">${state.done} / ${state.total}</span>
<a class="fs-ghostbtn" href="${api.labelExportUrl}" download>${icon("download", 14)}Export CSV</a>
</div>`;
}

function current() {
  const pair = state.queue[state.idx];
  const it = pair.item;
  const attrs = it.attributes || [];
  const gradeButtons = [0, 1, 2, 3].map((v) => `
<button class="fs-gbtn" aria-pressed="${state.marks[pair.pair_id] === v}" aria-keyshortcuts="${v}" data-grade="${v}"${state.saving ? " disabled" : ""}>
<span class="fs-gbtn-top"><span class="fs-dot d${v}"></span>${v}<span class="fs-key">${v}</span></span>
<span class="fs-gbtn-label">${GRADE_NAMES[v]}</span>
</button>`).join("");
  return `
<div><span class="fs-eyebrow">Query</span><p class="fs-qtext">${esc(pair.query)}</p></div>
<article class="fs-item text-only" aria-label="Item to grade">
<div class="fs-item-info">
<h2 class="fs-item-name">${esc(it.name)}</h2>
<p class="fs-meta" style="font-size: 15px">${esc(it.category_path)}</p>
${it.description ? `<p style="font-size: 15px">${esc(it.description)}</p>` : ""}
<p class="fs-price">${price(it.price)}${it.price > 0 ? ` <span class="fs-meta" style="font-weight: 400">· ${esc(it.price_bucket)}</span>` : ""}</p>
<div class="fs-chips">${attrs.map((a) => `<span class="fs-chip">${esc(a)}</span>`).join("")}</div>
</div>
</article>
<div class="fs-grades" role="group" aria-label="Grade this pair">${gradeButtons}</div>
${state.error ? `<p class="fs-note fs-error" role="alert">${esc(state.error)}</p>` : ""}
<div class="fs-row">
<button class="fs-ghostbtn" data-back${state.idx === 0 ? " disabled" : ""}><span class="fs-key">←</span>Back</button>
<span class="fs-meta">Pair ${state.idx + 1} of ${state.queue.length} in this batch · shuffled order</span>
<button class="fs-ghostbtn" data-skip>Skip<span class="fs-key">S</span></button>
</div>`;
}

function finished() {
  return `
<div class="fs-panel" style="display: flex; flex-direction: column; gap: 12px; align-items: flex-start; padding: 32px">
<h2>Batch complete</h2>
<p class="fs-meta" style="font-size: 15px">${Object.keys(state.marks).length} pairs graded this session. Export the CSV to compute weighted κ against the judge.</p>
<button class="fs-ghostbtn" data-restart>Review batch again</button>
</div>`;
}

function side() {
  const rubric = RUBRIC.map((r) => `<li><span class="fs-chip fs-grade g${r.g}"><span class="fs-dot"></span>${r.g} · ${r.label.toLowerCase()}</span><span>${esc(r.text)}</span></li>`).join("");
  const keyRow = (label, keys) => `<li style="flex-direction: row; align-items: center; justify-content: space-between; width: 100%"><span>${label}</span><span style="display: flex; gap: 4px">${keys}</span></li>`;
  return `
<aside class="fs-side">
<section class="fs-panel" aria-label="Rubric">
<button class="fs-disclosure" aria-expanded="${state.rubricOpen}" data-rubric>
<span class="fs-h2">Rubric</span>
<svg class="fs-chev${state.rubricOpen ? " open" : ""}" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"></path></svg>
</button>
${state.rubricOpen ? `<ul class="fs-rubric">${rubric}</ul>` : ""}
</section>
<section class="fs-panel" aria-label="Keyboard">
<h2>Keyboard</h2>
<ul class="fs-rubric" style="gap: 10px">
${keyRow("Grade", '<span class="fs-key">0</span><span class="fs-key">1</span><span class="fs-key">2</span><span class="fs-key">3</span>')}
${keyRow("Back", '<span class="fs-key">←</span>')}
${keyRow("Skip", '<span class="fs-key">S</span>')}
</ul>
</section>
<section class="fs-panel" aria-label="Blind protocol">
<h2>Blind by design</h2>
<p class="fs-meta" style="margin-top: 8px">System, rank, scores and the judge grade are hidden. Pairs are pooled from all systems and shown in shuffled order. The card is text only, the same evidence the LLM judge sees, so agreement compares like with like.</p>
</section>
</aside>`;
}

function go(ctx, i) {
  state.idx = Math.max(0, Math.min(state.queue.length, i));
  state.error = "";
  render(ctx);
}

async function submit(ctx, value) {
  if (state.saving || state.idx >= state.queue.length) return;
  const pair = state.queue[state.idx];
  state.saving = true;
  try {
    const res = await api.label(pair.pair_id, value);
    state.marks[pair.pair_id] = value;
    state.done = res.done;
    state.total = res.total;
    state.idx += 1;
    state.error = "";
  } catch (err) {
    state.error = `Not saved: ${err.message}`;
  }
  state.saving = false;
  render(ctx);
}

function onKey(ctx, event) {
  if (ctx.activeView() !== "label" || !state.queue || (event.target instanceof Element && event.target.matches("input, textarea"))) return;
  const k = event.key;
  if (["0", "1", "2", "3"].includes(k)) {
    event.preventDefault();
    submit(ctx, Number(k));
  } else if (k === "ArrowLeft") {
    event.preventDefault();
    go(ctx, state.idx - 1);
  } else if (k === "s" || k === "S") {
    event.preventDefault();
    go(ctx, state.idx + 1);
  }
}

let keysBound = false;

export async function render(ctx) {
  const root = ctx.root;
  if (!keysBound) {
    document.addEventListener("keydown", (e) => onKey(ctx, e));
    keysBound = true;
  }
  if (!state.queue) {
    root.innerHTML = '<p class="fs-loading" role="status">Loading label queue…</p>';
    try {
      const q = await api.labelQueue();
      state.queue = q.pairs;
      state.done = q.done;
      state.total = q.total;
    } catch (err) {
      root.innerHTML = `<p class="fs-note fs-error" role="alert">${esc(err.message)}</p>`;
      return;
    }
  }
  const main = state.idx < state.queue.length ? current() : finished();
  root.innerHTML = `
<div class="fs-main" style="margin-top: 0">
<section class="fs-labelmain" aria-label="Blind labeling">${progress()}${main}</section>
${side()}
</div>`;
  root.querySelectorAll("[data-grade]").forEach((b) => b.addEventListener("click", () => submit(ctx, Number(b.dataset.grade))));
  root.querySelector("[data-back]")?.addEventListener("click", () => go(ctx, state.idx - 1));
  root.querySelector("[data-skip]")?.addEventListener("click", () => go(ctx, state.idx + 1));
  root.querySelector("[data-restart]")?.addEventListener("click", () => go(ctx, 0));
  root.querySelector("[data-rubric]").addEventListener("click", () => {
    state.rubricOpen = !state.rubricOpen;
    render(ctx);
  });
}
