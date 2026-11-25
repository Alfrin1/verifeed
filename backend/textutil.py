"""
Pure-Python text helpers for VeriFeed (stdlib only, so they are easy to test):
language detection (Tamil / English / other), translation to English, keyword extraction,
claim-rating classification (English + Tamil + Hindi words) and Google News RSS parsing.
"""
import logging
import re
import xml.etree.ElementTree as ET

log = logging.getLogger("verifeed.text")


# ----------------------------------------------------------------------------- language
def _is_tamil(c):
    return "஀" <= c <= "௿"


def detect_language(text):
    """'en' | 'ta' | 'mixed' (Tamil + English) | 'other' (some other non-Latin script)."""
    chars = [c for c in (text or "") if c.isalpha() or _is_tamil(c)]
    if len(chars) < 3:
        return "en"
    n = len(chars)
    tamil = sum(1 for c in chars if _is_tamil(c))
    nonlatin = sum(1 for c in chars if not c.isascii())
    if tamil / n > 0.6:
        return "ta"
    if tamil / n > 0.1:
        return "mixed"
    if nonlatin / n > 0.3:
        return "other"
    return "en"


def translate_to_english(text, lang="auto"):
    """Best-effort translation. Returns '' if every method fails."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        from deep_translator import GoogleTranslator

        out = []
        for i in range(0, min(len(text), 9000), 4500):
            out.append(GoogleTranslator(source="auto", target="en").translate(text[i:i + 4500]) or "")
        res = " ".join(out).strip()
        if res:
            return res
    except Exception as e:
        log.warning("deep-translator failed: %s", type(e).__name__)
    if lang in ("ta", "mixed"):
        try:
            import httpx

            r = httpx.get(
                "https://api.mymemory.translated.net/get",
                params={"q": text[:450], "langpair": "ta|en"},
                timeout=8,
            )
            r.raise_for_status()
            res = ((r.json().get("responseData") or {}).get("translatedText") or "").strip()
            if res and not res.upper().startswith("MYMEMORY WARNING"):
                return res
        except Exception as e:
            log.warning("MyMemory translation failed: %s", type(e).__name__)
    return ""


# ----------------------------------------------------------------------------- tokens / keywords
STOP = set(
    "the and for that this with from have has was were are not you your will can his her they them their what when "
    "who why how about into over after before than then also just more most some any all out our get got its "
    "here there been being would could should very much many like only other new now one two "
    "இது அது ஒரு என்று மற்றும் உள்ள இந்த அந்த என தான் ஆனால் இல்லை என்ற ஆகும் உள்ளது".split()
)


def words(text):
    text = re.sub(r"https?://\S+", " ", (text or "").lower())
    text = re.sub(r"[#@]\S+", " ", text)  # hashtags / mentions are noise for matching
    return [w for w in re.split(r"[\s.,;:!?()\[\]{}\"'“”‘’\-–—/\\|…•*_~`+=<>]+", text) if w]


def stems(text):
    """Crude 5-character stems: lets inflected Tamil words and English plurals match."""
    return {w[:5] for w in words(text) if len(w) >= 3 and w not in STOP and not w.isdigit()}


def keyword_query(text, n=6):
    out, seen = [], set()
    for w in words(text):
        if len(w) < 3 or w in STOP or w.isdigit() or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= n:
            break
    return " ".join(out)


def sentences(text):
    parts = [s.strip() for s in re.split(r"[.!?।\n]+", text or "") if len(s.strip()) >= 25]
    if parts:
        return parts
    t = (text or "").strip()
    return [t[:200]] if len(t) >= 15 else []


# ----------------------------------------------------------------------------- sensationalism
SENSATIONAL = [
    "breaking", "shocking", "you won't believe", "exposed", "secret", "they don't want you",
    "share before", "urgent", "viral", "miracle", "cure", "banned", "leaked", "hoax", "wake up",
    "must watch", "100% proven", "conspiracy", "scam", "giveaway", "click the link", "forward this",
    "before it's deleted", "doctors hate", "government hiding", "mainstream media", "forward to everyone",
]
SENSATIONAL_TA = [
    "அதிர்ச்சி", "பகிருங்கள்", "பகிரவும்", "அவசரம்", "வைரல்", "எச்சரிக்கை", "பரபரப்பு", "ரகசியம்",
    "அம்பலம்", "அற்புத", "குணமாகும்", "குணப்படுத்த", "இலவசம்", "உடனே", "ஷேர்", "அனைவருக்கும் பகிர",
]


def sensationalism(original, english):
    low = ((original or "") + " " + (english or "")).lower()
    kw = sum(1 for k in SENSATIONAL + SENSATIONAL_TA if k in low)
    latin = [c for c in (english or "") if c.isascii() and c.isalpha()]
    caps = sum(c.isupper() for c in latin) / max(1, len(latin))
    score = 0.25 * kw
    if caps > 0.5 and len(latin) > 20:
        score += 0.3
    score += min(0.2, 0.05 * (original or "").count("!"))
    return min(1.0, score)


# ----------------------------------------------------------------------------- ratings
FALSE_WORDS = (
    "false", "fake", "incorrect", "misleading", "hoax", "scam", "pants on fire", "fabricated", "satire",
    "unproven", "no evidence", "manipulated", "altered", "wrong", "mislabel", "not true", "untrue", "debunk",
    "out of context", "doctored", "old video", "old photo", "unrelated", "morphed", "edited", "photoshopped",
    "deepfake", "ai-generated", "ai generated", "rumour", "rumor",
    # Tamil
    "போலி", "தவறான", "பொய்", "உண்மையல்ல", "உண்மை அல்ல", "திரிக்கப்பட்ட", "வதந்தி", "ஏமாற்று", "பழைய வீடியோ", "பழைய புகைப்படம்",
    # Hindi
    "फर्जी", "झूठ", "भ्रामक", "अफवाह",
)
TRUE_WORDS = ("true", "correct", "accurate", "legit", "genuine", "real", "உண்மை", "सच")


def classify_rating(text):
    low = (text or "").lower()
    if any(k in low for k in FALSE_WORDS):
        return "false"
    if any(k in low for k in TRUE_WORDS):
        return "true"
    return "mixed"


# ----------------------------------------------------------------------------- Google News RSS
def parse_rss(xml_text):
    out = []
    root = ET.fromstring(xml_text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src_el = it.find("source")
        src = (src_el.text or "").strip() if src_el is not None and src_el.text else ""
        src_url = src_el.get("url", "") if src_el is not None else ""
        if src and title.endswith(" - " + src):
            title = title[: -(len(src) + 3)]
        out.append({
            "title": title,
            "url": (it.findtext("link") or "").strip(),
            "date": (it.findtext("pubDate") or "").strip(),
            "source": src,
            "source_url": src_url,
        })
    return out
