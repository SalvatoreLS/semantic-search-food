import { esc, icon, imageSlot, wireImageFallbacks } from "./dom.js";
import { grade, price } from "./format.js";

const host = () => document.getElementById("fs-drawer-host");
let lastFocus = null;

export function closeDrawer() {
  host().innerHTML = "";
  document.removeEventListener("keydown", onEscape);
  lastFocus?.focus();
  lastFocus = null;
}

function onEscape(event) {
  if (event.key === "Escape") closeDrawer();
}

export function openDrawer(result, systemLabel) {
  lastFocus = document.activeElement;
  const g = grade(result.judge_grade);
  const attrs = result.attributes || [];
  const sources = result.sources || [];
  host().innerHTML = `
<button class="fs-scrim" aria-label="Close details" data-close></button>
<aside class="fs-drawer" aria-label="Item details" role="dialog" aria-modal="true">
<div class="fs-drawer-head">
<span class="fs-eyebrow">Result #${result.rank} · ${esc(systemLabel)}</span>
<button class="fs-iconbtn" aria-label="Close details" data-close>${icon("close")}</button>
</div>
<div class="fs-drawer-body">
<span class="fs-img">${imageSlot(result.item_id, "No image for this item", result.is_food, { size: 32 })}</span>
<div style="display: flex; flex-direction: column; gap: 8px">
<h3 class="fs-dname">${esc(result.name)}</h3>
<p class="fs-meta">${esc(result.category_path)} · ${price(result.price)} · ${esc(result.price_bucket)}</p>
<div class="fs-chips">
<span class="${g.cls}"><span class="fs-dot"></span>${g.label}</span>
${result.is_food ? "" : '<span class="fs-chip nonfood">non-food</span>'}
</div>
</div>
<section class="fs-section">
<span class="fs-eyebrow">Judge reason</span>
<p style="font-size: 15px">${esc(result.judge_reason || "Not judged for this query.")}</p>
</section>
<section class="fs-section">
<span class="fs-eyebrow">Retrieval sources</span>
<ol class="fs-srcs">${sources.map((s) => `<li><span>${esc(s.label)}</span><span class="fs-mono">#${s.rank}</span></li>`).join("")}</ol>
<span class="fs-meta fs-mono">final score ${Number(result.score).toFixed(3)}</span>
</section>
<section class="fs-section">
<span class="fs-eyebrow">Dietary attributes</span>
<div class="fs-chips">${attrs.length ? attrs.map((a) => `<span class="fs-chip">${esc(a)}</span>`).join("") : '<span class="fs-meta">None listed</span>'}</div>
</section>
<section class="fs-section">
<span class="fs-eyebrow">Indexed text</span>
<p class="fs-doc">${esc(result.doc_text)}</p>
</section>
<section class="fs-section">
<span class="fs-eyebrow">Item ID</span>
<div class="fs-idrow">
<code>${esc(result.item_id)}</code>
<button class="fs-ghostbtn" data-copy>${icon("copy", 14)}<span>Copy</span></button>
</div>
</section>
</div>
</aside>`;
  const root = host();
  wireImageFallbacks(root);
  root.querySelectorAll("[data-close]").forEach((b) => b.addEventListener("click", closeDrawer));
  root.querySelector("[data-copy]").addEventListener("click", async (event) => {
    const label = event.currentTarget.querySelector("span");
    try {
      await navigator.clipboard.writeText(result.item_id);
      label.textContent = "Copied";
    } catch (err) {
      label.textContent = "Copy failed";
    }
  });
  document.addEventListener("keydown", onEscape);
  root.querySelector(".fs-drawer [data-close]").focus();
}
