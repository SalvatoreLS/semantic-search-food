import { api } from "../api.js";
import { esc, icon, imageSlot, noteHtml, wireImageFallbacks } from "../dom.js";
import { MISSING, grade, isFoodQuery } from "../format.js";

const state = { text: "", left: "r0", right: "main", data: null, note: "", loading: false };
const STRIP = [["ndcg5", "nDCG@5", false], ["p5", "P@5", false], ["food_leak5", "food-leak@5", true]];

function header(ctx) {
  return `
<div class="fs-searchrow">
<div class="fs-search">
${icon("search", 20)}
<label for="fs-cq" class="fs-sr">Food query, shared by both columns</label>
<input id="fs-cq" type="text" value="${esc(state.text)}" placeholder="Descreva o que você quer comer…" autocomplete="off">
<button class="fs-enter" aria-label="Run query on both systems" data-run>${icon("enter", 18)}</button>
</div>
</div>
<div class="fs-chiprow">
<span class="fs-eyebrow">Eval queries</span>
<div class="fs-chipscroll">
${ctx.queries.map((q) => `<button class="fs-qchip" aria-pressed="${q.text === state.data?.query}" data-query="${esc(q.text)}">${esc(q.text)}</button>`).join("")}
</div>
</div>`;
}

function rankMap(results) {
  return new Map(results.map((r) => [r.item_id, r.rank]));
}

function column(ctx, side, sideData, otherData, foodQuery) {
  const sys = sideData.system;
  const otherSys = otherData.system;
  const otherRanks = rankMap(otherData.results);
  const short = ctx.systemShort(sys);
  const metrics = STRIP.map(([key, label, lowerIsBetter]) => {
    const v = sideData.metrics?.[key];
    const ov = otherData.metrics?.[key];
    const best = v != null && ov != null && (lowerIsBetter ? v < ov : v > ov);
    return `<div class="fs-mtile"><span class="fs-stat-label">${label}</span><span class="fs-mval${best ? " best" : ""}" title="${best ? `Better than ${esc(ctx.systemLabel(otherSys))}` : ""}">${v == null ? MISSING : v.toFixed(2)}</span></div>`;
  }).join("");
  const rows = sideData.results.map((r) => {
    const g = grade(r.judge_grade);
    const o = otherRanks.get(r.item_id);
    const dim = !r.is_food && foodQuery;
    return `
<li class="fs-li${o ? "" : " unique"}${dim ? " dim" : ""}">
<span class="fs-li-rank">${r.rank}</span>
<span class="fs-thumb">${imageSlot(r.item_id, r.name, r.is_food, { size: 18, withLabel: false })}</span>
<span class="fs-li-main"><span class="fs-li-name">${esc(r.name)}</span><span class="fs-li-sub">${esc(r.category_path)}${r.is_food ? "" : '<span class="fs-chip nonfood">non-food</span>'}</span></span>
<span class="fs-li-right">
<span class="${g.cls}"><span class="fs-dot"></span>${g.label}</span>
<span class="fs-overlap${o ? "" : " uniq"}"><span class="${o ? "fs-link" : "fs-ring"}"></span>${o ? `in both · #${r.rank} ↔ #${o}` : `only in ${esc(short)}`}</span>
</span>
</li>`;
  }).join("");
  const info = ctx.systems.find((s) => s.id === sys) || { label: sys, desc: "" };
  return `
<section aria-label="${esc(info.label)} results">
<div class="fs-colhead">
<div style="display: flex; flex-direction: column; gap: 2px; min-width: 0">
<span class="fs-eyebrow">${side === "left" ? "Left" : "Right"}</span>
<span class="fs-sysname">${esc(info.label)}</span>
<span class="fs-meta">${esc(info.desc)}</span>
</div>
<div class="fs-seg sm" role="group" aria-label="${side === "left" ? "Left" : "Right"} system">
${ctx.systems.map((s) => `<button aria-pressed="${s.id === sys}" data-side="${side}" data-system="${esc(s.id)}">${esc(s.short)}</button>`).join("")}
</div>
</div>
<div class="fs-mstrip">${metrics}</div>
<ol class="fs-list">${rows}</ol>
</section>`;
}

function body(ctx) {
  if (state.loading) return '<p class="fs-loading" role="status">Running query on both systems…</p>';
  const d = state.data;
  if (!d) return state.note ? "" : '<p class="fs-empty">Pick one of the 100 evaluation queries or type your own.</p>';
  const foodQuery = isFoodQuery(d);
  const rightRanks = rankMap(d.right.results);
  const overlap = d.left.results.filter((r) => rightRanks.has(r.item_id)).length;
  const n = Math.max(d.left.results.length, d.right.results.length);
  return `
<div class="fs-cmpsum">
<div style="display: flex; align-items: baseline; gap: 12px">
<span class="fs-eyebrow">Query</span>
<span style="font-size: 18px; font-weight: 600">${esc(d.query)}</span>
${d.intent ? `<span class="fs-chip intent">${esc(d.intent)}</span>` : ""}
</div>
<div style="display: flex; align-items: center; gap: 20px">
<span class="fs-overlap"><span class="fs-link"></span>${overlap} of ${n} in both lists</span>
<span class="fs-overlap uniq"><span class="fs-ring"></span>${n - overlap} unique per side</span>
</div>
</div>
<div class="fs-cmp">${column(ctx, "left", d.left, d.right, foodQuery)}${column(ctx, "right", d.right, d.left, foodQuery)}</div>`;
}

async function run(ctx, text) {
  const q = (text || "").trim();
  if (!q) return;
  state.text = q;
  state.loading = true;
  state.note = "";
  render(ctx);
  try {
    state.data = await api.compare(q, state.left, state.right);
  } catch (err) {
    state.data = null;
    state.note = err.status === 503 ? "Live query needs an API key; showing cached queries only." : err.message;
  }
  state.loading = false;
  render(ctx);
}

export function render(ctx) {
  const root = ctx.root;
  root.innerHTML = `${header(ctx)}${state.note ? noteHtml(state.note) : ""}${body(ctx)}`;
  wireImageFallbacks(root);
  const input = root.querySelector("#fs-cq");
  input.addEventListener("input", (e) => { state.text = e.target.value; });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") run(ctx, input.value); });
  root.querySelector("[data-run]").addEventListener("click", () => run(ctx, input.value));
  root.querySelectorAll("[data-query]").forEach((b) => b.addEventListener("click", () => run(ctx, b.dataset.query)));
  root.querySelectorAll("[data-side]").forEach((b) =>
    b.addEventListener("click", () => {
      state[b.dataset.side] = b.dataset.system;
      if (state.data) run(ctx, state.data.query);
    }),
  );
}
