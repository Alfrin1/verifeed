"""
VeriFeed backend (FastAPI) - Tamil + English
Pipeline:
  image  -> OCR (EasyOCR ta+en) + CV deepfake forensics (+Grad-CAM)
  text   -> language detect -> translate to English -> NLP risk
  claims -> Google Fact Check API (en+ta) + Indian fact-checker search (Newschecker, Fact Crescendo,
            YouTurn, BOOM, Alt News, Vishvas News, Factly via Google News RSS) + news corroboration
            (Google News RSS, optional NewsData.io / GNews)
  fusion -> weighted risk -> verdict JSON.   Redis cache (in-memory fallback), SQLite history.
Run:  uvicorn main:app --host 127.0.0.1 --port 8000
"""
import asyncio
import base64
import hashlib
import io
import logging
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline

from textutil import (
    classify_rating,
    detect_language,
    keyword_query,
    parse_rss,
    sensationalism,
    sentences,
    stems,
    translate_to_english,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("verifeed")

CV_MODEL = os.getenv("CV_MODEL", "dima806/deepfake_vs_real_image_detection")
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai-community/roberta-base-openai-detector")
FACT_KEY = os.getenv("GOOGLE_FACTCHECK_API_KEY", "")
NEWSDATA_KEY = os.getenv("NEWSDATA_API_KEY", "")  # optional, free tier 200 req/day
GNEWS_KEY = os.getenv("GNEWS_API_KEY", "")  # optional, free tier 100 req/day
OCR_LANGS = [x.strip() for x in os.getenv("OCR_LANGS", "ta,en").split(",") if x.strip()] or ["en"]
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL = int(os.getenv("CACHE_TTL", "86400"))
DB_PATH = os.getenv("DB_PATH", "verifeed_history.db")
FACT_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
ALLOWED_HOSTS = ("cdninstagram.com", "fbcdn.net", "fbsbx.com")
MAX_IMAGE_BYTES = 12 * 1024 * 1024
UA = {"User-Agent": "Mozilla/5.0 (VeriFeed research project)"}

# Indian fact-checkers searched through Google News (English + Tamil coverage)
FACTCHECK_SITES = [
    "newschecker.in", "factcrescendo.com", "youturn.in", "boomlive.in",
    "altnews.in", "vishvasnews.com", "factly.in",
]

# Paper (Sec. V-E) uses 0.45/0.35/0.20 for CV/NLP/fact-check. This build adds a news-corroboration signal,
# so the weights are re-balanced; missing signals are dropped and the rest re-normalised.
W_CV, W_NLP, W_FC, W_NEWS = 0.40, 0.30, 0.20, 0.10

M = {}  # loaded models
CV_LOCK = threading.Lock()  # Grad-CAM hooks are not thread-safe


def err(e):
    """Short error text that never contains URLs (API keys travel in query strings)."""
    if isinstance(e, httpx.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return type(e).__name__


# ----------------------------------------------------------------------------- cache
class Cache:
    def __init__(self):
        self.redis = None
        self.mem = {}

    async def init(self):
        try:
            import redis.asyncio as aioredis
            r = aioredis.from_url(REDIS_URL, decode_responses=True)
            await r.ping()
            self.redis = r
            log.info("Redis cache connected")
        except Exception as e:
            log.warning("Redis unavailable (%s) - using in-memory cache", e)

    async def get(self, key):
        import json
        if self.redis:
            try:
                v = await self.redis.get(key)
                return json.loads(v) if v else None
            except Exception:
                return None
        return self.mem.get(key)

    async def set(self, key, value):
        import json
        if self.redis:
            try:
                await self.redis.set(key, json.dumps(value), ex=CACHE_TTL)
                return
            except Exception:
                pass
        if len(self.mem) > 2000:
            self.mem.pop(next(iter(self.mem)))
        self.mem[key] = value


cache = Cache()


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, handle TEXT,"
        " image_url TEXT, caption TEXT, risk REAL, verdict TEXT)"
    )
    con.commit()
    con.close()


def save_history(handle, image_url, caption, risk, verdict):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO history (ts, handle, image_url, caption, risk, verdict) VALUES (?,?,?,?,?,?)",
        (time.time(), handle, image_url, caption[:500], risk, verdict),
    )
    con.commit()
    con.close()


# ----------------------------------------------------------------------------- models
def with_retry(fn, tries=4, delay=5):
    """Retry flaky network downloads (e.g. GitHub/HuggingFace 5xx timeouts)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            log.warning("Attempt %d/%d failed: %s", i + 1, tries, e)
            time.sleep(delay * (i + 1))
    raise last


def load_models():
    use_gpu = torch.cuda.is_available()
    log.info("Loading CV model %s", CV_MODEL)
    M["proc"] = with_retry(lambda: AutoImageProcessor.from_pretrained(CV_MODEL))
    M["cv"] = with_retry(lambda: AutoModelForImageClassification.from_pretrained(CV_MODEL)).eval()
    fake_idx = 1
    for i, label in M["cv"].config.id2label.items():
        if any(k in str(label).lower() for k in ("fake", "synthetic", "generated", "ai")):
            fake_idx = int(i)
            break
    M["fake_idx"] = fake_idx

    log.info("Loading OCR (EasyOCR, languages=%s)", OCR_LANGS)
    try:
        import easyocr
        try:
            M["ocr"] = with_retry(lambda: easyocr.Reader(OCR_LANGS, gpu=use_gpu), tries=2, delay=3)
            M["ocr_langs"] = OCR_LANGS
        except Exception as e:
            if OCR_LANGS == ["en"]:
                raise
            log.error("EasyOCR with %s failed (%s). Falling back to English-only OCR "
                      "(Tamil text inside images will NOT be read).", OCR_LANGS, e)
            M["ocr"] = with_retry(lambda: easyocr.Reader(["en"], gpu=use_gpu), tries=2, delay=3)
            M["ocr_langs"] = ["en"]
        M["ocr_engine"] = "easyocr"
    except Exception as e:
        log.error("EasyOCR could not be loaded (%s). Trying RapidOCR (English/Chinese only).", e)
        try:
            from rapidocr_onnxruntime import RapidOCR
            M["ocr"] = RapidOCR()
            M["ocr_engine"] = "rapidocr"
            M["ocr_langs"] = ["en"]
            log.info("Using RapidOCR")
        except Exception as e2:
            log.error("RapidOCR unavailable (%s). Continuing without OCR.", e2)

    log.info("Loading text model %s", TEXT_MODEL)
    M["nlp"] = with_retry(lambda: pipeline("text-classification", model=TEXT_MODEL, device=0 if use_gpu else -1))
    try:
        M["face"] = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception as e:
        M["face"] = None
        log.error("Face detector unavailable (%s). Fix: pip uninstall -y opencv-python opencv-python-headless "
                  "opencv-contrib-python && pip install \"opencv-python-headless<5\"", e)
    log.info("All models loaded")


# ----------------------------------------------------------------------------- CV + Grad-CAM
def has_face(img):
    if M.get("face") is None:
        return True  # detector unavailable: don't down-weight the CV signal
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = M["face"].detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    return len(faces) > 0


def _gradcam_target(model):
    try:
        return model.vit.encoder.layer[-1].layernorm_before
    except Exception:
        return None


def _make_heatmap(img, store):
    a = store["a"][0]
    g = store["g"][0]
    weights = g.mean(dim=0)
    cam = torch.relu((a * weights).sum(-1))[1:]  # drop CLS token
    n = int(math.sqrt(cam.numel()))
    if n * n != cam.numel():
        return None
    cam = cam.reshape(n, n).cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    small = img.convert("RGB").copy()
    small.thumbnail((640, 640))
    rgb = np.array(small)
    cam = cv2.resize(cam, (rgb.shape[1], rgb.shape[0]))
    colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * colored).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_cv(img):
    """Returns (fake_probability, heatmap_data_url_or_None)."""
    with CV_LOCK:
        model, proc = M["cv"], M["proc"]
        x = proc(images=img, return_tensors="pt")["pixel_values"]
        store = {}
        target = _gradcam_target(model)
        hook = None
        if target is not None:
            def fwd(mod, inp, out):
                store["a"] = out.detach()
                if out.requires_grad:
                    out.register_hook(lambda g: store.__setitem__("g", g.detach()))
            hook = target.register_forward_hook(fwd)
        heat = None
        try:
            with torch.enable_grad():
                logits = model(pixel_values=x).logits
                prob = torch.softmax(logits, dim=-1)[0, M["fake_idx"]].item()
                if hook is not None:
                    model.zero_grad()
                    logits[0, M["fake_idx"]].backward()
                    try:
                        if "a" in store and "g" in store:
                            heat = _make_heatmap(img, store)
                    except Exception as e:
                        log.warning("Grad-CAM failed: %s", e)
        finally:
            if hook is not None:
                hook.remove()
        return prob, heat


# ----------------------------------------------------------------------------- OCR
def run_ocr(img):
    text = ""
    try:
        arr = np.array(img.convert("RGB"))
        if M.get("ocr_engine") == "easyocr":
            parts = M["ocr"].readtext(arr, detail=0, paragraph=True)
        elif M.get("ocr_engine") == "rapidocr":
            res, _ = M["ocr"](arr[:, :, ::-1])  # RapidOCR expects BGR
            parts = [r[1] for r in (res or [])]
        else:
            raise RuntimeError("no OCR engine loaded")
        text = " ".join(parts).strip()
    except Exception as e:
        log.warning("OCR failed (%s), trying Tesseract", e)
        try:
            import pytesseract
            text = pytesseract.image_to_string(img).strip()
        except Exception:
            text = ""
    return re.sub(r"\s+", " ", text)[:1500]


# ----------------------------------------------------------------------------- NLP
def run_nlp(original, english, run_detector):
    """Returns (risk 0..1 or None, details). The AI-text detector only runs on genuine English text."""
    if len((original or "").strip()) < 15:
        return None, {}
    sens = sensationalism(original, english)
    ai_prob = 0.0
    if run_detector:
        try:
            res = M["nlp"](english[:2000], top_k=None, truncation=True, max_length=512)
            if res and isinstance(res[0], list):
                res = res[0]
            for r in res:
                if "fake" in r["label"].lower():
                    ai_prob = float(r["score"])
        except Exception as e:
            log.warning("NLP model failed: %s", e)
        score = 0.6 * sens + 0.4 * ai_prob
    else:
        score = sens  # translated text: the GPT-2 detector is meaningless, use sensationalism only
    return round(score, 4), {"sensationalism": round(sens, 3), "ai_text_prob": round(ai_prob, 3) if run_detector else None}


# ----------------------------------------------------------------------------- news / fact-check sources
async def google_news_rss(client, q, tamil=False):
    """Free, no key. Returns list of items, or None if the request failed."""
    hl, ceid = ("ta", "IN:ta") if tamil else ("en-IN", "IN:en")
    try:
        r = await client.get(
            "https://news.google.com/rss/search",
            params={"q": q, "hl": hl, "gl": "IN", "ceid": ceid},
            headers=UA,
            timeout=8,
        )
        r.raise_for_status()
        return parse_rss(r.text)[:20]
    except Exception as e:
        log.warning("Google News RSS failed: %s", err(e))
        return None


async def newsdata_search(client, q):
    try:
        r = await client.get(
            "https://newsdata.io/api/1/latest",
            params={"apikey": NEWSDATA_KEY, "q": q[:100], "language": "en,ta"},
            timeout=8,
        )
        r.raise_for_status()
        return [
            {"title": a.get("title") or "", "source": a.get("source_name") or a.get("source_id") or "",
             "url": a.get("link") or "", "date": a.get("pubDate") or "", "source_url": a.get("source_url") or ""}
            for a in (r.json().get("results") or [])
        ]
    except Exception as e:
        log.warning("NewsData.io failed: %s", err(e))
        return None


async def gnews_search(client, q):
    try:
        r = await client.get(
            "https://gnews.io/api/v4/search",
            params={"q": q, "lang": "en", "max": 10, "apikey": GNEWS_KEY},
            timeout=8,
        )
        r.raise_for_status()
        return [
            {"title": a.get("title") or "", "source": (a.get("source") or {}).get("name", ""),
             "url": a.get("url") or "", "date": a.get("publishedAt") or "",
             "source_url": (a.get("source") or {}).get("url", "")}
            for a in (r.json().get("articles") or [])
        ]
    except Exception as e:
        log.warning("GNews failed: %s", err(e))
        return None


def is_factcheck_source(item):
    blob = (item.get("source_url") or "") + " " + (item.get("url") or "")
    return any(d in blob for d in FACTCHECK_SITES)


async def site_factchecks(client, text, tamil):
    """Search the Indian fact-checker sites (via Google News) for articles about this claim."""
    q = keyword_query(text, 4)
    if not q:
        return []
    sites = " OR ".join("site:" + d for d in FACTCHECK_SITES)
    items = await google_news_rss(client, f"{q} ({sites})", tamil)
    qs = stems(q)
    need = max(1, min(2, len(qs)))
    out = []
    for it in items or []:
        if not any(d in (it.get("source_url") or "") for d in FACTCHECK_SITES):
            continue  # guard: only accept real fact-checker sources
        if len(qs & stems(it["title"])) < need:
            continue
        out.append({
            "claim": "", "publisher": it["source"], "title": it["title"][:160],
            "rating": "Fact-check article", "verdict": classify_rating(it["title"]),
            "url": it["url"], "via": "fact-checker search",
        })
    return out


async def run_factcheck(client, original, english, lang):
    """Returns (risk 0..1 or None, reviews). None => no matching fact-check found anywhere."""
    base = english or original
    if len(base.strip()) < 20 and len(original.strip()) < 20:
        return None, []
    tamil = lang in ("ta", "mixed")
    eng_ok = lang == "en" or (english and english != original)
    reviews, seen = [], set()

    def add(rv):
        if rv["url"] and rv["url"] not in seen:
            seen.add(rv["url"])
            reviews.append(rv)

    # --- 1) Google Fact Check API, English + Tamil queries
    g_queries = []
    if FACT_KEY:
        if eng_ok:
            g_queries += [(s[:200], "en") for s in sentences(base)[:2]]
            kq = keyword_query(base, 6)
            if kq:
                g_queries.append((kq, "en"))
        if lang != "en":
            lc = "ta" if tamil else None
            g_queries += [(s[:200], lc) for s in sentences(original)[:2]]
            okq = keyword_query(original, 6)
            if okq:
                g_queries.append((okq, lc))
        g_queries = list(dict.fromkeys(g_queries))

    async def google_one(q, lc):
        try:
            params = {"query": q, "key": FACT_KEY, "pageSize": 5}
            if lc:
                params["languageCode"] = lc
            r = await client.get(FACT_URL, params=params, timeout=8)
            r.raise_for_status()
            return q, r.json().get("claims", [])
        except Exception as e:
            log.warning("Google Fact Check failed: %s", err(e))
            return q, []

    # --- 2) Indian fact-checker sites (no key needed)
    site_jobs = []
    if eng_ok:
        site_jobs.append(site_factchecks(client, base, False))
    if lang != "en":
        site_jobs.append(site_factchecks(client, original, tamil))

    g_res, s_res = await asyncio.gather(
        asyncio.gather(*[google_one(q, lc) for q, lc in g_queries]),
        asyncio.gather(*site_jobs),
    )

    for q, claims in g_res:
        qs = stems(q)
        need = 2 if len(qs) <= 6 else 3
        for c in claims:
            if len(qs & stems(c.get("text", ""))) < need:  # Google search is loose; require real overlap
                continue
            for rv in c.get("claimReview", []):
                add({
                    "claim": c.get("text", "")[:200],
                    "publisher": (rv.get("publisher") or {}).get("name", ""),
                    "title": (rv.get("title") or "")[:160],
                    "rating": rv.get("textualRating", ""),
                    "verdict": classify_rating(rv.get("textualRating", "")),
                    "url": rv.get("url"),
                    "via": "Google Fact Check",
                })
    for lst in s_res:
        for rv in lst:
            add(rv)

    if not reviews:
        return None, []
    vals = [{"false": 1.0, "mixed": 0.5, "true": 0.0}[r["verdict"]] for r in reviews]
    score = 0.7 * max(vals) + 0.3 * (sum(vals) / len(vals))
    return round(score, 4), reviews[:8]


async def run_news(client, original, english, lang):
    """News corroboration. Returns (distinct_source_count or None if no source reachable, articles)."""
    if len(english.strip()) < 20 and len(original.strip()) < 20:
        return None, []
    tamil = lang in ("ta", "mixed")
    eng_ok = lang == "en" or (english and english != original)
    kq = keyword_query(english, 6) if eng_ok else ""
    kq5 = keyword_query(english, 5) if eng_ok else ""
    tq = keyword_query(original, 6) if tamil else ""

    jobs, queries = [], []
    if kq:
        jobs.append(google_news_rss(client, kq, False)); queries.append(kq)
    if tq:
        jobs.append(google_news_rss(client, tq, True)); queries.append(tq)
    if NEWSDATA_KEY and kq5:
        jobs.append(newsdata_search(client, kq5)); queries.append(kq5)
    if GNEWS_KEY and kq5:
        jobs.append(gnews_search(client, kq5)); queries.append(kq5)
    if not jobs:
        return None, []

    results = await asyncio.gather(*jobs)
    if all(r is None for r in results):
        return None, []

    arts, seen = [], set()
    for q, res in zip(queries, results):
        qs = stems(q)
        need = max(1, min(3, len(qs) // 2))
        for a in res or []:
            if not a["url"] or a["url"] in seen or is_factcheck_source(a):
                continue
            if len(qs & stems(a["title"])) < need:
                continue
            seen.add(a["url"])
            arts.append({"title": a["title"][:160], "source": a["source"], "url": a["url"], "date": a["date"]})
    sources = {a["source"].lower() for a in arts if a["source"]}
    return len(sources), arts[:6]


def news_score(n_sources, sens):
    """Coverage by several outlets lowers risk. No coverage only counts against SENSATIONAL claims."""
    if n_sources is None:
        return None
    if n_sources >= 3:
        return 0.15
    if n_sources >= 1:
        return 0.40
    return 0.65 if sens is not None and sens >= 0.25 else None


# ----------------------------------------------------------------------------- fusion
def fuse(cv, nlp, fc, news, face):
    parts = [(W_CV * (1.0 if face else 0.3), cv), (W_NLP, nlp), (W_FC, fc), (W_NEWS, news)]
    num = den = 0.0
    for w, s in parts:
        if s is not None:
            num += w * s
            den += w
    if den == 0:
        return None
    risk = num / den
    if fc is not None and fc >= 0.9:  # strong debunk overrides ambiguous image score (paper V-E)
        risk = max(risk, 0.75)
    return round(risk, 4)


def verdict_for(risk):
    if risk is None:
        return "Not enough signal", "gray"
    if risk < 0.35:
        return "Likely authentic", "green"
    if risk < 0.60:
        return "Suspicious", "orange"
    return "Likely manipulated / misleading", "red"


# ----------------------------------------------------------------------------- image fetch
def host_allowed(host):
    host = host or ""
    return any(host == d or host.endswith("." + d) for d in ALLOWED_HOSTS)


async def fetch_image(client, url):
    url = url.strip().strip('"').strip("'")
    p = urlparse(url)
    if p.scheme != "https" or not host_allowed(p.hostname):
        hint = ""
        if p.hostname and p.hostname.endswith("instagram.com") and "/p/" in p.path:
            hint = " This looks like a POST link, not an image. Right-click the photo itself, or open the image in a new tab and copy that address."
        raise HTTPException(
            400,
            f"Image host not allowed: scheme={p.scheme!r} host={p.hostname!r}. "
            f"Expected an https address on cdninstagram.com or fbcdn.net.{hint}",
        )
    r = await client.get(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.instagram.com/"},
        timeout=15,
    )
    r.raise_for_status()
    if len(r.content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image too large")
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    img.thumbnail((1280, 1280))
    return img


async def guarded(fn, *args, default=None):
    try:
        return await asyncio.to_thread(fn, *args)
    except Exception:
        log.exception("%s failed", getattr(fn, "__name__", "task"))
        return default


async def safe(coro, default):
    try:
        return await coro
    except Exception:
        log.exception("task failed")
        return default


# ----------------------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(load_models)
    await cache.init()
    init_db()
    app.state.http = httpx.AsyncClient(follow_redirects=False)  # image downloads: never follow redirects
    app.state.api = httpx.AsyncClient(follow_redirects=True)  # fact-check / news APIs
    yield
    await app.state.http.aclose()
    await app.state.api.aclose()


app = FastAPI(title="VeriFeed", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AnalyzeRequest(BaseModel):
    image_url: str = ""
    caption: str = ""
    handle: str = ""


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "fact_check_enabled": bool(FACT_KEY),
        "news_sources": {"google_news_rss": True, "newsdata": bool(NEWSDATA_KEY), "gnews": bool(GNEWS_KEY)},
        "ocr_engine": M.get("ocr_engine"),
        "ocr_langs": M.get("ocr_langs"),
        "redis": cache.redis is not None,
    }


@app.post("/api/v1/analyze")
async def analyze(req: AnalyzeRequest):
    t0 = time.perf_counter()
    caption = (req.caption or "").strip()[:4000]
    path = urlparse(req.image_url).path if req.image_url else ""
    key = "vf2:" + hashlib.md5(path.encode()).hexdigest() + ":" + hashlib.md5(caption.encode()).hexdigest()

    hit = await cache.get(key)
    if hit:
        hit["cached"] = True
        hit["latency_ms"] = int((time.perf_counter() - t0) * 1000)
        return hit

    client = app.state.http
    api = app.state.api
    img = None
    if req.image_url:
        try:
            img = await fetch_image(client, req.image_url)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Could not download image: {err(e)}")

    cv_score, heatmap, ocr_text, face = None, None, "", False
    if img is not None:
        (cv_res, ocr_text, face) = await asyncio.gather(
            guarded(run_cv, img, default=(None, None)),
            guarded(run_ocr, img, default=""),
            guarded(has_face, img, default=False),
        )
        cv_score, heatmap = cv_res

    combined = (caption + "\n" + ocr_text).strip()
    lang = detect_language(combined)
    english = combined
    notes = []
    if lang != "en":
        tr = await guarded(translate_to_english, combined, lang, default="")
        if tr:
            english = tr
        else:
            notes.append("Translation to English failed: text analysis used the original language only.")

    nlp_res, fc_res, news_res = await asyncio.gather(
        guarded(run_nlp, combined, english, lang == "en", default=(None, {})),
        safe(run_factcheck(api, combined, english, lang), (None, [])),
        safe(run_news(api, combined, english, lang), (None, [])),
    )
    nlp_score, nlp_detail = nlp_res
    fc_score, reviews = fc_res
    n_sources, articles = news_res
    nws = news_score(n_sources, (nlp_detail or {}).get("sensationalism"))

    risk = fuse(cv_score, nlp_score, fc_score, nws, face)
    verdict, color = verdict_for(risk)

    if img is None:
        notes.append("No image available (video or text-only post): only the caption was analysed.")
    elif not face:
        notes.append("No face detected: the deepfake model is face-oriented, so its weight was reduced.")
    if lang in ("ta", "mixed"):
        notes.append("Tamil text detected and translated to English for analysis.")
    if img is not None and lang == "en" and "ta" not in (M.get("ocr_langs") or []):
        notes.append("OCR is English-only on this server, so Tamil text inside images is not read.")
    if not FACT_KEY:
        notes.append("Google Fact Check key not set: only Indian fact-checker search was used.")
    if fc_score is None and combined:
        notes.append("No matching published fact-check found (this is not proof the claim is true).")
    if n_sources is None and combined:
        notes.append("News sources could not be reached.")
    elif n_sources == 0 and nws is not None:
        notes.append("Sensational claim with no matching coverage from news outlets.")

    out = {
        "integrity_score": None if risk is None else round((1 - risk) * 100),
        "risk": risk,
        "verdict": verdict,
        "color": color,
        "language": lang,
        "translated_text": english if lang != "en" and english != combined else "",
        "signals": {"cv": cv_score, "nlp": nlp_score, "fact_check": fc_score, "news": nws},
        "nlp_detail": nlp_detail,
        "ocr_text": ocr_text,
        "fact_checks": reviews,
        "news_articles": articles,
        "heatmap": heatmap,
        "notes": notes,
        "cached": False,
    }
    if risk is not None:
        await cache.set(key, dict(out))
    asyncio.create_task(asyncio.to_thread(save_history, req.handle, req.image_url, caption, risk, verdict))
    out["latency_ms"] = int((time.perf_counter() - t0) * 1000)
    return out
