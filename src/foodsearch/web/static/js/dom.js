const ESCAPES = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };

export function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

function svg(size, body) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
}

const PATHS = {
  search: '<circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.3-4.3"></path>',
  enter: '<polyline points="9 10 4 15 9 20"></polyline><path d="M20 4v7a4 4 0 0 1-4 4H4"></path>',
  shuffle: '<path d="m18 14 4 4-4 4"></path><path d="m18 2 4 4-4 4"></path><path d="M2 18h1.4c1.3 0 2.5-.6 3.3-1.7l6.1-8.6c.7-1.1 2-1.7 3.3-1.7H22"></path><path d="M2 6h1.4c1.3 0 2.5.6 3.3 1.7l.6.8"></path><path d="M22 18h-6.1c-1.3 0-2.5-.6-3.3-1.7l-.6-.8"></path>',
  info: '<circle cx="12" cy="12" r="10"></circle><path d="M12 16v-4"></path><path d="M12 8h.01"></path>',
  chevron: '<path d="m6 9 6 6 6-6"></path>',
  utensils: '<path d="M3 2v7c0 1.1.9 2 2 2h4a2 2 0 0 0 2-2V2"></path><path d="M7 2v20"></path><path d="M21 15V2a5 5 0 0 0-5 5v6c0 1.1.9 2 2 2h3Zm0 0v7"></path>',
  box: '<path d="m7.5 4.27 9 5.15"></path><path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"></path><path d="m3.3 7 8.7 5 8.7-5"></path><path d="M12 22V12"></path>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"></rect><circle cx="9" cy="9" r="2"></circle><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"></path>',
  close: '<path d="M18 6 6 18"></path><path d="m6 6 12 12"></path>',
  copy: '<rect x="8" y="8" width="14" height="14" rx="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line>',
  arrow: '<path d="m9 18 6-6-6-6"></path>',
};

export function icon(name, size = 16) {
  return svg(size, PATHS[name]);
}

export function imageSlot(itemId, name, isFood, { size = 28, label = name, withLabel = true } = {}) {
  const fallback = `<span class="fs-fallback">${icon(isFood ? "utensils" : "box", size)}${withLabel ? `<span>${esc(label)}</span>` : ""}</span>`;
  return `${fallback}<img src="/images/${encodeURIComponent(itemId)}" alt="" loading="lazy" data-fallback>`;
}

export function wireImageFallbacks(root) {
  root.querySelectorAll("img[data-fallback]").forEach((img) => {
    const drop = () => img.remove();
    if (img.complete && img.naturalWidth === 0) drop();
    else img.addEventListener("error", drop, { once: true });
  });
}

const segThumbs = new Map();

function addThumb(container) {
  const thumb = document.createElement("span");
  thumb.className = "fs-seg-thumb";
  thumb.setAttribute("aria-hidden", "true");
  container.prepend(thumb);
  return thumb;
}

function moveThumb(thumb, target) {
  thumb.style.transform = `translateX(${target.offsetLeft}px)`;
  thumb.style.width = `${target.offsetWidth}px`;
}

function jumpThumb(thumb, target) {
  thumb.classList.remove("anim");
  moveThumb(thumb, target);
  thumb.getBoundingClientRect();
  thumb.classList.add("anim");
}

function followResize(container, thumb, selected) {
  let width = container.offsetWidth;
  new ResizeObserver(() => {
    const target = selected();
    if (!target || !container.isConnected || container.offsetWidth === width) return;
    width = container.offsetWidth;
    jumpThumb(thumb, target);
  }).observe(container);
}

export function wireSegs(root) {
  root.querySelectorAll(".fs-seg").forEach((seg) => {
    const key = seg.getAttribute("aria-label");
    const thumb = addThumb(seg);
    const place = (btn) => {
      moveThumb(thumb, btn);
      segThumbs.set(key, { x: btn.offsetLeft, w: btn.offsetWidth });
    };
    const pressed = () => seg.querySelector('button[aria-pressed="true"]');
    const prev = segThumbs.get(key);
    const target = pressed();
    if (!target) return;
    if (prev) {
      thumb.style.transform = `translateX(${prev.x}px)`;
      thumb.style.width = `${prev.w}px`;
      thumb.getBoundingClientRect();
      thumb.classList.add("anim");
      place(target);
    } else {
      jumpThumb(thumb, target);
      place(target);
    }
    followResize(seg, thumb, pressed);
    seg.querySelectorAll("button").forEach((btn) =>
      btn.addEventListener("click", () => {
        if (!seg.isConnected) return;
        seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
        moveThumb(thumb, btn);
        setTimeout(() => { if (seg.isConnected) place(btn); });
      }),
    );
  });
}

export function syncNavThumb(nav) {
  const current = () => nav.querySelector('[aria-current="page"]');
  const target = current();
  if (!target) return;
  const thumb = nav.querySelector(".fs-seg-thumb");
  if (thumb) {
    moveThumb(thumb, target);
    return;
  }
  const created = addThumb(nav);
  jumpThumb(created, target);
  followResize(nav, created, current);
}

export function noteHtml(text) {
  return `<p class="fs-note" role="status">${icon("info", 14)}${esc(text)}</p>`;
}

export function skel(cls = "") {
  return `<span class="fs-skel${cls ? ` ${cls}` : ""}"></span>`;
}

export function loadingStatus(html) {
  return `
<div class="fs-progress" aria-hidden="true"><span></span></div>
<p class="fs-loadline" role="status"><span class="fs-status fs-pulse"></span><span>${html}</span><span class="fs-elapsed" aria-hidden="true" data-elapsed></span></p>`;
}

export function createLoader(delayMs = 150) {
  let seq = 0;
  let delay = 0;
  let tick = 0;
  const stop = () => {
    clearTimeout(delay);
    clearInterval(tick);
  };
  return {
    start(root, show) {
      stop();
      const id = ++seq;
      const t0 = performance.now();
      const update = () => {
        const el = root.querySelector("[data-elapsed]");
        if (el) el.textContent = `${((performance.now() - t0) / 1000).toFixed(1)} s`;
      };
      delay = setTimeout(() => {
        show();
        update();
        tick = setInterval(update, 100);
      }, delayMs);
      return () => {
        if (id !== seq) return false;
        stop();
        return true;
      };
    },
  };
}
