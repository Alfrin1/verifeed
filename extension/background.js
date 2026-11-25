// Service worker: relays analyze requests to the FastAPI backend (avoids page CORS / mixed-content limits).
const DEFAULT_API = "http://127.0.0.1:8000";

async function apiBase() {
  const { apiBase } = await chrome.storage.sync.get("apiBase");
  return (apiBase || DEFAULT_API).replace(/\/$/, "");
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type !== "ANALYZE") return;
  (async () => {
    try {
      const res = await fetch((await apiBase()) + "/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(msg.payload),
      });
      if (!res.ok) throw new Error("Backend returned " + res.status);
      sendResponse({ ok: true, data: await res.json() });
    } catch (e) {
      sendResponse({ ok: false, error: String(e.message || e) });
    }
  })();
  return true; // keep channel open for async response
});
