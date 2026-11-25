"""
VeriFeed evaluation runner  (standalone - does NOT modify or start the live server)

Imports the functions already defined in main.py and runs a labelled dataset through the
SAME pipeline that /api/v1/analyze uses (CV + Grad-CAM, OCR, translation, NLP, fact-check,
news, fusion). Results are saved per sample to results/eval_results.csv so plots can be
regenerated at any time with make_plots.py without re-running the models.

Usage (run from anywhere; this file lives in <project>/eval/):
    python eval/eval_run.py                       # full run (uses internet for fact-check/news)
    python eval/eval_run.py --offline             # skip fact-check + news (fast, reproducible)
    python eval/eval_run.py --limit 20            # quick smoke test on 20 samples
    python eval/eval_run.py --resume              # continue an interrupted run

Dataset: eval/dataset/labels.csv  (see README.md)
    id,image_path,caption,label,category
    label: 1 = fake / manipulated / misleading, 0 = authentic
"""
import argparse
import asyncio
import csv
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # project folder that contains main.py and textutil.py
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # so .env (API keys) and relative paths resolve exactly like the server

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

FIELDS = [
    "id", "image_path", "caption", "label", "category",
    "cv", "nlp", "fact_check", "news", "face", "risk", "verdict", "language",
    "cold_ms", "warm_ms", "n_fact_checks", "n_news", "error",
]


def read_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["id"] = (r.get("id") or str(i + 1)).strip()
        r["image_path"] = (r.get("image_path") or "").strip()
        r["caption"] = (r.get("caption") or "").strip()
        r["category"] = (r.get("category") or "").strip()
        if str(r.get("label", "")).strip() not in ("0", "1"):
            raise SystemExit(f"Row {r['id']}: label must be 0 or 1, got {r.get('label')!r}")
        r["label"] = int(r["label"])
    return rows


def load_image(rel):
    if not rel:
        return None
    p = Path(rel)
    if not p.is_absolute():
        p = HERE / "dataset" / rel  # paths are relative to eval/dataset/
    img = Image.open(p).convert("RGB")
    img.thumbnail((1280, 1280))  # same preprocessing as fetch_image()
    return img


async def analyze_local(vf, api, img, caption, offline):
    """Mirror of analyze() in main.py, minus the HTTP download / cache / SQLite steps."""
    cv_score, heat, ocr_text, face = None, None, "", False
    if img is not None:
        cv_res, ocr_text, face = await asyncio.gather(
            vf.guarded(vf.run_cv, img, default=(None, None)),
            vf.guarded(vf.run_ocr, img, default=""),
            vf.guarded(vf.has_face, img, default=False),
        )
        cv_score, heat = cv_res

    combined = (caption + "\n" + ocr_text).strip()
    lang = vf.detect_language(combined)
    english = combined
    if lang != "en":
        tr = await vf.guarded(vf.translate_to_english, combined, lang, default="")
        if tr:
            english = tr

    if offline:
        nlp_res = await vf.guarded(vf.run_nlp, combined, english, lang == "en", default=(None, {}))
        fc_res, news_res = (None, []), (None, [])
    else:
        nlp_res, fc_res, news_res = await asyncio.gather(
            vf.guarded(vf.run_nlp, combined, english, lang == "en", default=(None, {})),
            vf.safe(vf.run_factcheck(api, combined, english, lang), (None, [])),
            vf.safe(vf.run_news(api, combined, english, lang), (None, [])),
        )
    nlp_score, nlp_detail = nlp_res
    fc_score, reviews = fc_res
    n_sources, articles = news_res
    nws = vf.news_score(n_sources, (nlp_detail or {}).get("sensationalism"))
    risk = vf.fuse(cv_score, nlp_score, fc_score, nws, face)
    verdict, color = vf.verdict_for(risk)
    out = {
        "risk": risk, "verdict": verdict, "color": color, "language": lang,
        "signals": {"cv": cv_score, "nlp": nlp_score, "fact_check": fc_score, "news": nws},
        "heatmap": heat, "fact_checks": reviews, "news_articles": articles, "ocr_text": ocr_text,
    }
    return out, face


async def main_async(args):
    import main as vf  # noqa: E402  (imports models code only; the server/lifespan is NOT started)

    logging.getLogger("verifeed").setLevel(logging.WARNING)
    rows = read_labels(HERE / "dataset" / "labels.csv")
    if args.limit:
        rows = rows[: args.limit]
    out_path = HERE / "results" / "eval_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.resume and out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            done = {r["id"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(done)} samples already done")
    mode = "a" if (args.resume and out_path.exists()) else "w"

    print("Loading models (same as server start-up)...")
    await asyncio.to_thread(vf.load_models)
    await vf.cache.init()
    api = httpx.AsyncClient(follow_redirects=True)
    print(f"OCR: {vf.M.get('ocr_engine')} {vf.M.get('ocr_langs')} | Redis: {vf.cache.redis is not None} "
          f"| fact-check key: {bool(vf.FACT_KEY)} | offline: {args.offline}")

    # Warm-up pass (not recorded): the first inference pays one-time lazy-init costs.
    todo = [r for r in rows if r["id"] not in done]
    if todo and not args.no_warmup:
        try:
            print("Warm-up run (discarded)...")
            await analyze_local(vf, api, load_image(todo[0]["image_path"]), todo[0]["caption"], True)
        except Exception as e:
            print("warm-up skipped:", type(e).__name__)

    with open(out_path, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            w.writeheader()
        for n, r in enumerate(todo, 1):
            row = {k: "" for k in FIELDS}
            row.update(id=r["id"], image_path=r["image_path"], caption=r["caption"].replace("\n", " "),
                       label=r["label"], category=r["category"])
            try:
                img = load_image(r["image_path"])
                # ---- COLD: full pipeline, cache bypassed
                t0 = time.perf_counter()
                res, face = await analyze_local(vf, api, img, r["caption"], args.offline)
                row["cold_ms"] = round((time.perf_counter() - t0) * 1000, 1)

                # ---- WARM: same cache set/get path the server uses (Redis if up, else memory)
                key = "evalrun:" + hashlib.md5((r["image_path"] + r["caption"]).encode()).hexdigest()
                await vf.cache.set(key, dict(res))
                t1 = time.perf_counter()
                hit = await vf.cache.get(key)
                row["warm_ms"] = round((time.perf_counter() - t1) * 1000, 3) if hit else ""
                if vf.cache.redis is not None:
                    try:
                        await vf.cache.redis.delete(key)  # leave no eval keys in your Redis
                    except Exception:
                        pass
                else:
                    vf.cache.mem.pop(key, None)

                s = res["signals"]
                row.update(cv=s["cv"], nlp=s["nlp"], fact_check=s["fact_check"], news=s["news"],
                           face=int(bool(face)), risk=res["risk"], verdict=res["verdict"],
                           language=res["language"], n_fact_checks=len(res["fact_checks"]),
                           n_news=len(res["news_articles"]))
            except Exception as e:  # keep going; bad sample is recorded, not fatal
                row["error"] = f"{type(e).__name__}: {e}"[:200]
            w.writerow(row)
            fh.flush()
            print(f"[{n}/{len(todo)}] id={row['id']} label={row['label']} risk={row['risk']} "
                  f"cold={row['cold_ms']}ms {row['error']}")
    await api.aclose()
    print(f"\nSaved -> {out_path}\nNext: python eval/make_plots.py")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="only first N samples")
    ap.add_argument("--offline", action="store_true", help="skip fact-check and news lookups")
    ap.add_argument("--resume", action="store_true", help="append to existing results, skipping done ids")
    ap.add_argument("--no-warmup", action="store_true")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
