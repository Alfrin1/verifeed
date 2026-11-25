// VeriFeed content script: MutationObserver over the Instagram feed, passive zero-click analysis.
(() => {
  const seen = new WeakSet();
  const queue = [];
  let running = 0;
  const MAX_PARALLEL = 2;

  function pickImage(article) {
    let best = null, area = 0;
    for (const img of article.querySelectorAll("img")) {
      const src = img.currentSrc || img.src;
      if (!src || !/cdninstagram|fbcdn/.test(src)) continue;
      if (img.closest("header")) continue; // skip avatar
      const a = (img.naturalWidth || img.width) * (img.naturalHeight || img.height);
      if (a > area) { area = a; best = src; }
    }
    return area >= 150 * 150 ? best : null;
  }

  function pickHandle(article) {
    const a = article.querySelector("header a[href^='/']");
    return a ? a.getAttribute("href").replace(/\//g, "") : "";
  }

  function pickCaption(article) {
    const handle = pickHandle(article);
    let best = "";
    for (const el of article.querySelectorAll("h1, span[dir='auto'], div[dir='auto']")) {
      let t = (el.innerText || "").trim();
      if (!t || el.closest("header")) continue;
      if (handle && t.startsWith(handle)) t = t.slice(handle.length).trim();
      if (t.length > best.length && t.length > 15 && !/^\d[\d,.]*\s*(likes?|views?|comments?)/i.test(t)) best = t;
    }
    return best.replace(/\s*more$/i, "");
  }

  function mountBadge(article) {
    let badge = article.querySelector(":scope .vf-badge");
    if (badge) return badge;
    badge = document.createElement("div");
    badge.className = "vf-badge vf-gray";
    badge.textContent = "VeriFeed: analysing…";
    const header = article.querySelector("header");
    if (header && header.parentElement) header.parentElement.insertBefore(badge, header.nextSibling);
    else article.prepend(badge);
    return badge;
  }

  function esc(s) {
    const d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function showModal(d) {
    document.querySelector(".vf-overlay")?.remove();
    const ov = document.createElement("div");
    ov.className = "vf-overlay";
    const s = d.signals || {};
    const pct = (v) => (v == null ? "n/a" : Math.round(v * 100) + "%");
    const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "#");
    const checks = (d.fact_checks || []).map((f) =>
      `<li><b>${esc(f.publisher)}</b>: ${esc(f.rating)} &mdash; <a href="${esc(safeUrl(f.url))}" target="_blank" rel="noopener noreferrer">${esc(f.title || f.claim)}</a></li>`).join("");
    const news = (d.news_articles || []).map((a) =>
      `<li><b>${esc(a.source)}</b> &mdash; <a href="${esc(safeUrl(a.url))}" target="_blank" rel="noopener noreferrer">${esc(a.title)}</a></li>`).join("");
    ov.innerHTML = `
      <div class="vf-modal">
        <button class="vf-close" aria-label="Close">&times;</button>
        <h2>${esc(d.verdict)} ${d.integrity_score == null ? "" : "(integrity " + d.integrity_score + "/100)"}</h2>
        <table>
          <tr><td>Image forensics (fake probability)</td><td>${pct(s.cv)}</td></tr>
          <tr><td>Caption / text risk</td><td>${pct(s.nlp)}</td></tr>
          <tr><td>Fact-check risk</td><td>${pct(s.fact_check)}</td></tr>
          <tr><td>News-coverage risk</td><td>${pct(s.news)}</td></tr>
        </table>
        ${d.language && d.language !== "en" ? `<p class="vf-note">Language detected: ${esc(d.language === "ta" ? "Tamil" : d.language === "mixed" ? "Tamil + English" : d.language)}</p>` : ""}
        ${d.translated_text ? `<p><b>English translation:</b> ${esc(d.translated_text.slice(0, 400))}</p>` : ""}
        ${d.heatmap ? `<p>Grad-CAM heatmap (regions that drove the image verdict):</p><img class="vf-heat" src="${d.heatmap}" alt="Grad-CAM heatmap">` : ""}
        ${d.ocr_text ? `<p><b>Text found in image:</b> ${esc(d.ocr_text)}</p>` : ""}
        ${checks ? `<p><b>Published fact-checks:</b></p><ul>${checks}</ul>` : ""}
        ${news ? `<p><b>Related news coverage:</b></p><ul>${news}</ul>` : ""}
        ${(d.notes || []).map((n) => `<p class="vf-note">${esc(n)}</p>`).join("")}
        <p class="vf-note">${d.cached ? "Cached result. " : ""}${d.latency_ms} ms. Automated estimate, not a definitive ruling.</p>
      </div>`;
    ov.addEventListener("click", (e) => {
      if (e.target === ov || e.target.classList.contains("vf-close")) ov.remove();
    });
    document.body.appendChild(ov);
  }

  function render(badge, d) {
    badge.className = "vf-badge vf-" + (d.color || "gray");
    badge.textContent = "VeriFeed: " + d.verdict + (d.integrity_score == null ? "" : " · " + d.integrity_score + "/100") + "  (details)";
    badge.onclick = () => showModal(d);
  }

  function runNext() {
    while (running < MAX_PARALLEL && queue.length) {
      const job = queue.shift();
      running++;
      chrome.runtime.sendMessage({ type: "ANALYZE", payload: job.payload }, (resp) => {
        running--;
        if (chrome.runtime.lastError || !resp || !resp.ok) {
          job.badge.className = "vf-badge vf-gray";
          job.badge.textContent = "VeriFeed: backend unreachable (is the server running?)";
        } else {
          render(job.badge, resp.data);
        }
        runNext();
      });
    }
  }

  function scan(article) {
    if (seen.has(article)) return;
    const image_url = pickImage(article);
    const caption = pickCaption(article);
    if (!image_url && !caption) return; // not loaded yet; observer will fire again
    seen.add(article);
    const badge = mountBadge(article);
    queue.push({ badge, payload: { image_url: image_url || "", caption, handle: pickHandle(article) } });
    runNext();
  }

  function scanAll() {
    document.querySelectorAll("article").forEach(scan);
  }

  let timer = null;
  new MutationObserver(() => {
    clearTimeout(timer);
    timer = setTimeout(scanAll, 600); // debounce to protect scroll performance
  }).observe(document.body, { childList: true, subtree: true });

  scanAll();
})();
