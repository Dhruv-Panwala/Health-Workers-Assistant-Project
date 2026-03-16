const rawBaseUrl = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").trim();

export function buildApiUrl(path = "/query") {
  const base = rawBaseUrl.endsWith("/") ? rawBaseUrl.slice(0, -1) : rawBaseUrl;
  const suffix = path.startsWith("/") ? path : `/${path}`;
  return `${base}${suffix}`;
}

export const API_QUERY_URL = buildApiUrl("/query");
