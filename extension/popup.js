const api = document.getElementById("api");
const s = document.getElementById("s");
chrome.storage.sync.get("apiBase", (v) => { api.value = v.apiBase || "http://127.0.0.1:8000"; });
document.getElementById("save").onclick = () =>
  chrome.storage.sync.set({ apiBase: api.value.trim() }, () => (s.textContent = "Saved."));
document.getElementById("test").onclick = async () => {
  try {
    const r = await fetch(api.value.trim().replace(/\/$/, "") + "/health");
    const j = await r.json();
    s.textContent = "Backend OK. Fact-check: " + (j.fact_check_enabled ? "on" : "OFF (no API key)") + ", Redis: " + (j.redis ? "on" : "off (memory cache)");
  } catch (e) {
    s.textContent = "Cannot reach backend.";
  }
};
