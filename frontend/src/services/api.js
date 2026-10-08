// All network access lives here. The React app contains NO prediction/model logic:
// it sends the user's answers to FastAPI and renders what comes back.
// VITE_API_URL wins. Otherwise: dev server -> local FastAPI on :8000; built app -> same origin
// (FastAPI can serve the build itself).
const BASE = (import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "")).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, { errors = [], network = false, status = 0 } = {}) {
    super(message);
    this.errors = errors;
    this.network = network;
    this.status = status;
  }
}

async function request(path, options = {}, timeoutMs = 20000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...options,
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      signal: ctrl.signal,
    });
  } catch {
    throw new ApiError(
      "We couldn't reach the EarlyDia server. Please check that the backend is running and try again.",
      { network: true }
    );
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    /* non-JSON body */
  }
  if (!res.ok) {
    throw new ApiError(data?.message || "Something went wrong. Please try again.", {
      errors: Array.isArray(data?.errors) ? data.errors : [],
      status: res.status,
    });
  }
  return data;
}

export const getConfig = () => request("/api/config");
export const getHealth = () => request("/api/health");
export const predict = (payload) =>
  request("/api/predict", { method: "POST", body: JSON.stringify(payload) }, 30000);

export const getResearch = (name) => request(`/api/research/${name}`);
