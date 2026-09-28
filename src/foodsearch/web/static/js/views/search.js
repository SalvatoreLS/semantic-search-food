import { api } from "../api.js";
import { esc, icon, imageSlot, noteHtml, wireImageFallbacks, wireSegs } from "../dom.js";
import { openDrawer } from "../drawer.js";
import { cost, grade, isFoodQuery, ms, priceLine, priorText, sourcesShort, tiles } from "../format.js";

const state = { text: "", system: "main", traceOpen: false, data: null, note: "", loading: false };

function searchBar(ctx) {
  return `
<div class="fs-searchrow">
<div class="fs-search">
${icon("search", 20)}
<label for="fs-q" class="fs-sr">Food query</label>
<input id="fs-q" type="text" value="${esc(state.text)}" placeholder="Descreva o que você quer comer…" autocomplete="off">
<button class="fs-enter" aria-label="Run query" data-run>${icon("enter", 18)}</button>
</div>
<div class="fs-sysfield">
<span class="fs-eyebrow" aria-hidden="true">System</span>
<div class="fs-seg" role="group" aria-label="System">
${ctx.systems.map((s) => `<button aria-pressed="${s.id === state.system}" data-system="${esc(s.id)}">${esc(s.label)}</button>`).join("")}
</div>
</div>
</div>
<div class="fs-chiprow">
<span class="fs-eyebrow">Eval queries</span>
<div class="fs-chipscroll">
${ctx.queries.map((q) => `<button class="fs-qchip" aria-pressed="${q.text === state.data?.query}" data-query="${esc(q.text)}">${esc(q.text)}</button>`).join("")}
</div>
<button class="fs-ghostbtn" data-random>${icon("shuffle", 14)}Random query</button>
</div>`;
}

function understandingPanel(d) {
  if (!d.intent) {
    const text = d.system === "bm25"
      ? "BM25 has no query understanding: it matches stemmed tokens of the raw query."
      : "This system embeds the raw query directly: no intent, no dish expansion, no food prior.";
    return `<section class="fs-panel" aria-label="Query understanding"><div class="fs-row"><h2>Query understanding</h2></div><p class="fs-meta" style="margin-top: 12px">${text}</p></section>`;
  }
  const prior = priorText(d.food_prior);
  const dishes = d.expanded_dishes || [];
  return `
<section class="fs-panel" aria-label="Query understanding">
<div class="fs-row"><h2>Query understanding</h2>${d.understanding_model ? `<span class="fs-chip mono muted">${esc(d.understanding_model)}</span>` : ""}</div>
<dl class="fs-dl">
<div class="fs-dl-row"><dt>Intent</dt><dd><span class="fs-chip intent">${esc(d.intent)}</span></dd></div>
<div class="fs-dl-row"><dt>Expanded dishes</dt><dd class="fs-chips">
${dishes.length ? dishes.map((t) => `<span class="fs-chip muted">${esc(t)}</span>`).join("") : '<span class="fs-meta">None: this system sets the intent with a keyword rule and has no LLM expansion.</span>'}
</dd></div>
<div class="fs-dl-row"><dt>Food prior</dt><dd style="display: flex; flex-direction: column; gap: 4px"><span class="fs-prior"><span class="${prior.dot}"></span>${esc(prior.text)}</span><span class="fs-meta fs-mono">${esc(d.food_prior?.reason || "")}</span></dd></div>
</dl>
</section>`;
}

function tracePanel(d) {
  const stages = d.stages || [];
  const totalMs = stages.reduce((a, s) => a + (s.ms || 0), 0);
  const totalCost = stages.reduce((a, s) => a + (s.cost_usd || 0), 0);
  const steps = stages
    .map((s) => {
      const llm = (s.model || "").startsWith("gpt");
      return `<li class="fs-step"><span class="fs-node${llm ? " llm" : ""}"></span><span class="fs-step-text"><span class="fs-step-name">${esc(s.name)}</span><span class="fs-step-sub">${esc(s.detail || s.model || "deterministic")}</span></span><span class="fs-step-nums"><span>${ms(s.ms)}</span><span>${cost(s.cost_usd)}</span></span></li>`;
    })
    .join("");
  return `
<section class="fs-panel" aria-label="Pipeline trace">
<button class="fs-disclosure" aria-expanded="${state.traceOpen}" data-trace>
<span style="display: flex; flex-direction: column; gap: 2px"><span class="fs-h2">Pipeline trace</span><span class="fs-meta fs-mono">${stages.length} steps · ${ms(totalMs)} · ${cost(totalCost)}</span></span>
<svg class="fs-chev${state.traceOpen ? " open" : ""}" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"></path></svg>
</button>
${state.traceOpen ? `<ol class="fs-steps">${steps}<li class="fs-step"><span class="fs-node end"></span><span class="fs-step-text"><span class="fs-step-name">Top 10</span><span class="fs-step-sub">returned to UI</span></span><span class="fs-step-nums"></span></li></ol><div class="fs-legend"><span><span class="fs-node llm" style="margin: 0"></span>LLM call</span><span><span class="fs-node" style="margin: 0"></span>no LLM</span></div>` : ""}
</section>`;
}

function metricsPanel(d, judgeModel) {
  const t = tiles(d.metrics, d.system);
  if (!t.length) return "";
  return `
<section class="fs-panel" aria-label="Query metrics">
<div class="fs-row"><h2>Query metrics</h2><span class="fs-meta">judge: ${esc(judgeModel)}</span></div>
<div class="fs-stats">${t.map((x) => `<div class="fs-stat"><span class="fs-stat-label">${x.label}</span><span class="fs-stat-val">${x.value}</span><span class="${x.cls}">${x.delta}</span></div>`).join("")}</div>
</section>`;
}

function card(r, i, foodQuery) {
  const g = grade(r.judge_grade);
  const dim = !r.is_food && foodQuery;
  return `
<button class="fs-card${dim ? " dim" : ""}" style="animation-delay: ${i * 20}ms" data-rank="${r.rank}" aria-label="Details for ${esc(r.name)}">
<span class="fs-img">${imageSlot(r.item_id, r.name, r.is_food)}<span class="fs-rank">#${r.rank}</span></span>
<span class="fs-body">
<span class="fs-name">${esc(r.name)}</span>
<span class="fs-meta">${esc(r.category_path)}</span>
<span class="fs-meta">${esc(priceLine(r.price, r.price_bucket))}</span>
<span class="fs-chips">
<span class="${g.cls}"><span class="fs-dot"></span>${g.label}</span>
${r.is_food ? "" : '<span class="fs-chip nonfood">non-food</span>'}
<span class="fs-chip mono muted">${esc(sourcesShort(r.sources))}</span>
</span>
<span class="fs-score">score ${Number(r.score).toFixed(3)}</span>
</span>
</button>`;
}

function body(ctx) {
  if (state.loading) return '<p class="fs-loading" role="status">Running query…</p>';
  const d = state.data;
  if (!d) return state.note ? "" : '<p class="fs-empty">Pick one of the 100 evaluation queries or type your own.</p>';
  const sysLabel = ctx.systemLabel(d.system);
  const foodQuery = isFoodQuery(d);
  return `
<div class="fs-main">
<aside class="fs-side">${understandingPanel(d)}${tracePanel(d)}${metricsPanel(d, ctx.judgeModel)}</aside>
<section class="fs-results" aria-label="Results">
<div class="fs-reshead"><h2 class="fs-h2">Top 10 <span class="fs-meta" style="font-weight: 400">· ${esc(sysLabel)}</span></h2><span class="fs-meta">Select a card for details</span></div>
<div class="fs-grid">${d.results.map((r, i) => card(r, i, foodQuery)).join("")}</div>
</section>
</div>`;
}

async function run(ctx, text) {
  const q = (text || "").trim();
  if (!q) return;
  state.text = q;
  state.loading = true;
  state.note = "";
  render(ctx);
  try {
    state.data = await api.search(q, state.system);
  } catch (err) {
    state.data = null;
    state.note = err.status === 503 ? "Live query needs an API key; showing cached queries only." : err.message;
  }
  state.loading = false;
  render(ctx);
}

export function render(ctx) {
  const root = ctx.root;
  root.innerHTML = `${searchBar(ctx)}${state.note ? noteHtml(state.note) : ""}${body(ctx)}`;
  wireImageFallbacks(root);
  wireSegs(root);
  const input = root.querySelector("#fs-q");
  input.addEventListener("input", (e) => { state.text = e.target.value; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(ctx, input.value); });
  root.querySelector("[data-run]").addEventListener("click", () => run(ctx, input.value));
  root.querySelectorAll("[data-system]").forEach((b) =>
    b.addEventListener("click", () => {
      state.system = b.dataset.system;
      if (state.data) run(ctx, state.data.query);
      else render(ctx);
    }),
  );
  root.querySelectorAll("[data-query]").forEach((b) => b.addEventListener("click", () => run(ctx, b.dataset.query)));
  root.querySelector("[data-random]").addEventListener("click", () => {
    const pool = ctx.queries.filter((q) => q.text !== state.data?.query);
    if (pool.length) run(ctx, pool[Math.floor(Math.random() * pool.length)].text);
  });
  root.querySelector("[data-trace]")?.addEventListener("click", () => {
    state.traceOpen = !state.traceOpen;
    render(ctx);
  });
  root.querySelectorAll("[data-rank]").forEach((b) =>
    b.addEventListener("click", () => {
      const r = state.data.results.find((x) => String(x.rank) === b.dataset.rank);
      if (r) openDrawer(r, ctx.systemLabel(state.data.system));
    }),
  );
}
