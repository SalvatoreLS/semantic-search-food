import { api } from "../api.js";
import { esc, icon } from "../dom.js";
import { MISSING } from "../format.js";

const KEYS = [["ndcg5", false], ["ndcg10", false], ["p5", false], ["food_leak5", true]];

function box(title, sub, llm = false) {
  return `<div class="fs-abox${llm ? " llm" : ""}"><strong>${esc(title)}</strong><span>${esc(sub)}</span></div>`;
}

function chain(boxes) {
  return boxes.join(`<span class="fs-arrow">${icon("arrow")}</span>`);
}

function table(ctx, s) {
  const best = {};
  KEYS.forEach(([k, lower]) => {
    const vals = s.rows.map((r) => r[k]?.[0]).filter((v) => v != null);
    best[k] = lower ? Math.min(...vals) : Math.max(...vals);
  });
  const rows = s.rows.map((r) => {
    const info = ctx.systems.find((x) => x.id === r.system) || { label: r.system, desc: "" };
    const cells = KEYS.map(([k]) => {
      const v = r[k];
      if (!v) return `<td>${MISSING}</td>`;
      const ci = v[1] == null || v[2] == null ? "" : `<span class="fs-ci">[${v[1].toFixed(2)}, ${v[2].toFixed(2)}]</span>`;
      return `<td><span class="${v[0] === best[k] ? "fs-best" : ""}">${v[0].toFixed(2)}</span>${ci}</td>`;
    }).join("");
    return `<tr><td><span style="display: flex; flex-direction: column; gap: 2px"><span style="font-weight: 600">${esc(info.label)}</span><span class="fs-meta" style="font-weight: 400">${esc(info.desc)}</span></span></td>${cells}</tr>`;
  }).join("") || `<tr><td colspan="5"><span class="fs-meta">No metrics yet: they appear after <code>foodsearch judge --subset all</code> and <code>foodsearch eval</code>.</span></td></tr>`;
  return `<table class="fs-table"><thead><tr><th scope="col">System</th><th scope="col">nDCG@5</th><th scope="col">nDCG@10</th><th scope="col">P@5</th><th scope="col">food-leak@5 ↓</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function stat(label, value, sub) {
  return `<div class="fs-stat"><span class="fs-stat-label">${esc(label)}</span><span class="fs-stat-val">${esc(value)}</span><span class="fs-delta flat">${esc(sub)}</span></div>`;
}

export async function render(ctx) {
  const root = ctx.root;
  root.innerHTML = '<p class="fs-loading" role="status">Loading summary…</p>';
  let s;
  try {
    s = await api.summary();
  } catch (err) {
    root.innerHTML = `<p class="fs-note fs-error" role="alert">${esc(err.message)}</p>`;
    return;
  }
  const judge = s.judge || {};
  const kappa = judge.kappa == null ? MISSING : `κ ${judge.kappa.toFixed(2)}`;
  root.innerHTML = `
<div style="display: flex; justify-content: space-between; align-items: flex-end">
<div style="display: flex; flex-direction: column; gap: 4px">
<span class="fs-eyebrow">Headline results</span>
<h1 style="font-size: 32px; font-weight: 600; letter-spacing: -0.01em">${s.n_queries} eval queries, mean with 95% CI</h1>
</div>
<span class="fs-meta">Judge grades 0–3 · nDCG uses graded gains</span>
</div>
<div style="display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); column-gap: 24px; align-items: start">
<section style="grid-column: span 8; border: 1px solid var(--border); border-radius: 12px; overflow: hidden" aria-label="Systems by metric">${table(ctx, s)}</section>
<aside style="grid-column: span 4; display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px">
${stat("Total LLM cost", s.cost_usd == null ? MISSING : `$${s.cost_usd.toFixed(2)}`, "whole project")}
${stat("Judge vs human", kappa, `weighted, ${judge.model || MISSING}`)}
${stat("Pairs judged", (judge.judged ?? 0).toLocaleString("en-US"), `${judge.unjudged ?? 0} unjudged`)}
${stat("Human labels", `${s.label?.done ?? 0} / ${s.label?.total ?? 0}`, "blind, shuffled")}
</aside>
</div>
<section class="fs-panel" aria-label="Architecture" style="display: flex; flex-direction: column; gap: 16px">
<div class="fs-row"><h2>Architecture · Hybrid (headline)</h2><span class="fs-legend" style="margin: 0"><span><span class="fs-node llm" style="margin: 0"></span>LLM call</span><span><span class="fs-node" style="margin: 0"></span>no LLM</span></span></div>
<div class="fs-arch">${chain([
    box("Query (PT)", "free text or eval query"),
    box("Query understanding", "intent + dish expansion · gpt-4.1-mini", true),
    box("Retrieval", "dense raw + expanded, BM25 on expanded"),
    box("Weighted RRF", "dense 1 + 1, BM25 0.5"),
    box("Food prior", "non-food × λ, off for product intent"),
    box("Listwise rerank", "top 30 · gpt-4.1-mini", true),
    box("Top 10", "to UI"),
  ])}</div>
<div class="fs-arch" style="margin-top: 8px">${chain([
    box("LLM judge", `grades each query–item pair 0–3 · ${judge.model || "gpt-4.1"}`, true),
    box("Metrics", "nDCG@5/10, P@5, food-leak@5 per system"),
    box("Human validation", "blind labels, weighted κ vs judge"),
  ])}</div>
</section>`;
}
