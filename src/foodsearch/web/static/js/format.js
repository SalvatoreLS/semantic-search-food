export const MISSING = "n/a";

export const GRADE_LABELS = { 3: "perfect", 2: "good", 1: "partial", 0: "irrelevant" };

export const RUBRIC = [
  { g: 3, label: "Perfect", text: "Matches the core intent and its key modifiers. A customer would order it for this query." },
  { g: 2, label: "Good", text: "Right dish or kind of food, but misses a modifier. A reasonable substitute." },
  { g: 1, label: "Partial", text: "Related: ingredient, side, grocery or frozen version, or right cuisine but wrong meal." },
  { g: 0, label: "Irrelevant", text: "Unrelated food, or any non-food item for a food query." },
];

export const QUERY_METRICS = [
  ["ndcg5", "nDCG@5", false],
  ["ndcg10", "nDCG@10", false],
  ["p5", "P@5", false],
  ["food_leak5", "food-leak@5", true],
];

export function price(value) {
  return value == null ? MISSING : `R$ ${Number(value).toFixed(2).replace(".", ",")}`;
}

export function cost(value) {
  if (value == null) return MISSING;
  return value === 0 ? "$0" : `$${Number(Number(value).toPrecision(2))}`;
}

export function ms(value) {
  return value == null ? "" : `${Math.round(value)} ms`;
}

export function grade(g) {
  const unjudged = g == null;
  return {
    cls: `fs-chip fs-grade ${unjudged ? "gx" : `g${g}`}`,
    label: unjudged ? "not judged" : `${g} · ${GRADE_LABELS[g]}`,
  };
}

const SOURCE_ABBREV = { "dense raw": "raw", "dense expanded": "exp" };

export function sourcesShort(sources) {
  return (sources || [])
    .filter((s) => !s.label.includes("rerank"))
    .map((s) => `${SOURCE_ABBREV[s.label] || s.label} ${s.rank}`)
    .join(" · ");
}

export function isFoodQuery(d) {
  const kind = d.intent ?? d.query_type;
  return kind !== "product";
}

export function tiles(metrics, systemId, referenceId = "r0") {
  const mine = metrics?.[systemId];
  const ref = metrics?.[referenceId];
  if (!mine) return [];
  return QUERY_METRICS.map(([key, label, lowerIsBetter]) => {
    const value = mine[key];
    const tile = { label, value: value == null ? MISSING : value.toFixed(2) };
    if (systemId === referenceId) return { ...tile, delta: "reference", cls: "fs-delta flat" };
    if (value == null || ref?.[key] == null) return { ...tile, delta: "", cls: "fs-delta flat" };
    const d = Math.round((value - ref[key]) * 100) / 100;
    if (d === 0) return { ...tile, delta: "±0.00 vs R0", cls: "fs-delta flat" };
    const good = lowerIsBetter ? d < 0 : d > 0;
    return {
      ...tile,
      delta: `${d > 0 ? "▲ +" : "▼ −"}${Math.abs(d).toFixed(2)} vs R0`,
      cls: `fs-delta ${good ? "good" : "bad"}`,
    };
  });
}

export function priorText(prior) {
  if (!prior) return { text: "n/a", dot: "fs-status off" };
  return prior.on
    ? { text: `on (λ = ${prior.lambda})`, dot: "fs-status" }
    : { text: "off: product intent", dot: "fs-status off" };
}
