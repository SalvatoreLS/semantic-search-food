import { api } from "./api.js";
import { esc, syncNavThumb } from "./dom.js";
import { closeDrawer } from "./drawer.js";
import * as about from "./views/about.js";
import * as compare from "./views/compare.js";
import * as label from "./views/label.js";
import * as search from "./views/search.js";

const VIEWS = { search, compare, label, about };
const THEME_KEY = "fs-theme";
const root = document.getElementById("fs-root");

const ctx = {
  queries: [],
  systems: [],
  judgeModel: "gpt-4.1",
  systemLabel: (id) => ctx.systems.find((s) => s.id === id)?.label || id,
  systemShort: (id) => ctx.systems.find((s) => s.id === id)?.short || id,
  activeView: () => currentView(),
};

function readTheme() {
  try {
    return localStorage.getItem(THEME_KEY);
  } catch (err) {
    return null;
  }
}

function applyTheme(theme) {
  root.dataset.theme = theme === "dark" ? "dark" : "light";
  try {
    localStorage.setItem(THEME_KEY, root.dataset.theme);
  } catch (err) {
    return;
  }
}

function parseHash() {
  const [name, query = ""] = location.hash.replace("#", "").split("?");
  return { name: name in VIEWS ? name : "search", params: new URLSearchParams(query) };
}

function currentView() {
  return parseHash().name;
}

function route() {
  closeDrawer();
  const { name, params } = parseHash();
  document.querySelectorAll(".fs-view").forEach((el) => { el.hidden = el.dataset.view !== name; });
  document.querySelectorAll(".fs-nav a").forEach((a) => {
    if (a.dataset.view === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  syncNavThumb(document.querySelector(".fs-nav"));
  const view = VIEWS[name];
  const viewCtx = { ...ctx, root: document.getElementById(`view-${name}`) };
  if (params.get("q") && view.open) view.open(viewCtx, params);
  else view.render(viewCtx);
}

async function boot() {
  applyTheme(readTheme() || "light");
  document.getElementById("fs-theme").addEventListener("click", () =>
    applyTheme(root.dataset.theme === "dark" ? "light" : "dark"),
  );
  try {
    const meta = await api.queries();
    ctx.queries = meta.queries;
    ctx.systems = meta.systems;
    ctx.judgeModel = meta.judge_model || ctx.judgeModel;
  } catch (err) {
    document.getElementById("view-search").hidden = false;
    document.getElementById("view-search").innerHTML = `<p class="fs-note fs-error" role="alert">${esc(err.message)}</p>`;
    return;
  }
  window.addEventListener("hashchange", route);
  route();
}

boot();
