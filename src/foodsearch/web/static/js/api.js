export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { headers: { Accept: "application/json" }, ...options });
  } catch (err) {
    throw new ApiError(0, "Backend unreachable. Start it with `foodsearch serve`.");
  }
  if (!response.ok) {
    let detail = "";
    try {
      detail = (await response.json()).detail;
    } catch (err) {
      detail = response.statusText;
    }
    throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return response.json();
}

function qs(params) {
  return new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== "")).toString();
}

export const api = {
  queries: () => request("/api/queries"),
  search: (q, system) => request(`/api/search?${qs({ q, system })}`),
  compare: (q, left, right) => request(`/api/compare?${qs({ q, left, right })}`),
  labelQueue: () => request("/api/label/queue"),
  label: (pairId, grade) =>
    request("/api/label", {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify({ pair_id: pairId, grade }),
    }),
  labelExportUrl: "/api/label/export",
  summary: () => request("/api/summary"),
};
