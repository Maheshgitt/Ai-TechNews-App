"""
pipeline.py  v4
───────────────
Fixes:
  • Fetches 5–8+ articles reliably (multi-page fetch + relaxed filter)
  • Perplexity-style summary format
  • Smart push: only notifies when articles are NEW vs last run
  • 24-hr article cache with timestamp
"""

import re, json, time, hashlib, logging, os, requests
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY   = os.getenv("NEWSDATA_API_KEY")
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
FCM_SERVER_KEY = os.getenv("FCM_SERVER_KEY")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH", "/etc/secrets/service-account.json")

client = Groq(api_key=GROQ_API_KEY)
if not NEWS_API_KEY: raise ValueError("NEWSDATA_API_KEY missing")
if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing")

BASE_DIR    = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "memory" / "seen_articles.json"
PREV_HASHES_FILE = BASE_DIR / "memory" / "prev_hashes.json"
LOG_DIR     = BASE_DIR / "logs"
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"run_{datetime.now():%Y-%m-%d}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("pipeline")

# ── tunables ──────────────────────────────────────────────
MAX_FETCH       = 50   # articles fetched per API call
TARGET_ARTICLES = 8    # final articles to return (5–8 range)
MIN_SCORE       = 3    # lowered from previous to catch more articles
DEDUP_RATIO     = 0.72
MEMORY_EXPIRY_DAYS = 1  # 24-hour memory (matches app's 24hr display window)
GROQ_RETRY      = 3
GROQ_DELAY      = 2

# ── keywords ──────────────────────────────────────────────
KEYWORD_WEIGHTS: dict[str, int] = {
    "openai": 10, "gpt-5": 10, "gpt-4o": 10, "o3": 10, "o4": 10,
    "chatgpt": 9, "dall-e": 8, "sora": 9,
    "claude": 10, "anthropic": 10,
    "gemini": 10, "google deepmind": 10,
    "grok": 9, "xai": 9,
    "llama": 9, "meta ai": 9, "mistral": 9,
    "deepseek": 10, "qwen": 9,
    "perplexity": 8, "cohere": 7,
    "stability ai": 8, "midjourney": 8, "runway": 8,
    "large language model": 10, "llm": 10,
    "multimodal": 9, "vlm": 9,
    "agi": 10, "artificial general intelligence": 10,
    "reasoning model": 9,
    "generative ai": 9, "foundation model": 9,
    "ai agent": 9, "agentic ai": 9,
    "rag": 8, "fine-tuning": 8,
    "transformer": 7, "neural network": 7,
    "deep learning": 7, "machine learning": 7,
    "computer vision": 7, "nlp": 7,
    "text to image": 7, "text to video": 8,
    "diffusion model": 8,
    "copilot": 8, "github copilot": 9,
    "cursor": 7, "hugging face": 8, "langchain": 7,
    "nvidia": 8, "amd": 7, "intel": 7, "apple silicon": 8,
    "qualcomm": 7, "tsmc": 8,
    "gpu": 7, "cpu": 7, "tpu": 8, "npu": 8, "asic": 7,
    "fpga": 6, "chip": 5, "processor": 6,
    "semiconductor": 7, "2nm": 9, "3nm": 8,
    "quantum computing": 9, "quantum chip": 9,
    "humanoid robot": 9, "boston dynamics": 8,
    "figure ai": 9, "self-driving": 8, "waymo": 8,
    "robot": 6, "robotics": 6, "automation": 5,
    "zero-day": 10, "cve": 8, "ransomware": 8,
    "cybersecurity": 7, "vulnerability": 7, "exploit": 8,
    "data breach": 8,
    "5g": 5, "6g": 7, "starlink": 6,
    "ai regulation": 8, "ai safety": 9, "ai act": 8,
    "breakthrough": 5, "open-source": 5, "benchmark": 5,
    # lower tier — still count
    "artificial intelligence": 6, "technology": 2, "startup": 3,
    "google": 4, "microsoft": 4, "apple": 4, "amazon": 4, "meta": 4,
}

REJECT_PATTERN = re.compile(
    r"\b(stock price|share price|quarterly earnings|revenue beat|ipo filing"
    r"|market cap|shares surge|dividend|fiscal year|analyst rating"
    r"|cfo resigns|layoffs count)\b",
    re.IGNORECASE,
)

SOURCE_SCORES: dict[str, int] = {
    "techcrunch.com": 10, "theverge.com": 10, "wired.com": 10,
    "arstechnica.com": 10, "ieee.org": 10, "nature.com": 10,
    "venturebeat.com": 8, "zdnet.com": 8, "thenextweb.com": 8,
    "engadget.com": 8, "tomshardware.com": 8,
    "reuters.com": 7, "bloomberg.com": 7,
    "openai.com": 9, "anthropic.com": 9,
    "buzzfeed.com": -6, "dailymail.co.uk": -6,
}


# ══════════════════════════════════════════════════════════
# MEMORY  (24-hr window)
# ══════════════════════════════════════════════════════════
class ArticleMemory:
    def __init__(self, path: Path = MEMORY_FILE):
        self.path = path
        self._data: dict[str, str] = {}
        self._load()
        self._purge()

    def _load(self):
        if self.path.exists():
            try: self._data = json.loads(self.path.read_text())
            except: self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def _purge(self):
        cutoff = datetime.now() - timedelta(days=MEMORY_EXPIRY_DAYS)
        self._data = {h: ts for h, ts in self._data.items()
                      if datetime.fromisoformat(ts) > cutoff}
        self._save()

    @staticmethod
    def _hash(title: str) -> str:
        return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]

    def seen(self, title: str) -> bool:
        return self._hash(title) in self._data

    def mark_batch(self, titles: list[str]):
        now = datetime.now().isoformat()
        for t in titles: self._data[self._hash(t)] = now
        self._save()

    def size(self) -> int: return len(self._data)


# ══════════════════════════════════════════════════════════
# FETCH  (multi-page to guarantee enough articles)
# ══════════════════════════════════════════════════════════
def fetch_news() -> list[dict]:
    """Fetch up to MAX_FETCH articles across multiple categories."""
    all_articles = []
    categories = ["technology", "science"]

    for cat in categories:
        try:
            r = requests.get(
                "https://newsdata.io/api/1/news",
                params={"apikey": NEWS_API_KEY, "category": cat,
                        "language": "en", "size": 10},
                timeout=15,
            )
            r.raise_for_status()
            articles = r.json().get("results", [])
            all_articles.extend(articles)
            log.info(f"Fetched {len(articles)} from category={cat}")
        except Exception as e:
            log.error(f"Fetch error ({cat}): {e}")

    log.info(f"Total raw articles: {len(all_articles)}")
    return all_articles


# ══════════════════════════════════════════════════════════
# PRE-FILTER
# ══════════════════════════════════════════════════════════
def _kw_score(title: str, desc: str) -> int:
    text = (title + " " + (desc or "")).lower()
    return sum(w for kw, w in KEYWORD_WEIGHTS.items()
               if re.search(r"\b" + re.escape(kw) + r"\b", text))

def _src_score(article: dict) -> int:
    url = (article.get("source_url") or article.get("link") or "").lower()
    for domain, bonus in SOURCE_SCORES.items():
        if domain in url: return bonus
    return 0

def _is_dup(new_title: str, seen: list[str]) -> bool:
    return any(SequenceMatcher(None, new_title.lower(), s.lower()).ratio() >= DEDUP_RATIO
               for s in seen)

def prefilter(articles: list[dict], memory: ArticleMemory) -> list[dict]:
    scored = []
    seen_titles: list[str] = []
    for a in articles:
        title = (a.get("title") or "").strip()
        desc  = (a.get("description") or "").strip()
        if not title or memory.seen(title) or REJECT_PATTERN.search(title):
            continue
        if _is_dup(title, seen_titles):
            continue
        total = _kw_score(title, desc) + _src_score(a)
        if total >= MIN_SCORE:          # relaxed threshold
            seen_titles.append(title)
            scored.append((total, a))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [a for _, a in scored[:25]]
    log.info(f"Pre-filter: {len(candidates)} candidates (min_score={MIN_SCORE}).")
    return candidates


# ══════════════════════════════════════════════════════════
# LLM CLASSIFIER
# ══════════════════════════════════════════════════════════
CLASSIFIER_SYSTEM = f"""
You are a strict tech news classifier for software/hardware engineers.

Select the top {TARGET_ARTICLES} most impactful articles from the list.
Prioritise: AI model releases, hardware, robotics, cybersecurity, open-source.
Reject: pure finance, celebrity, generic PR.

Return ONLY valid JSON — no markdown, no preamble:
{{"selected": [1, 2, 5, ...]}}
""".strip()

def llm_classify(candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    numbered = "\n\n".join(
        f"{i}. {a.get('title','')}\n   {(a.get('description') or '')[:200]}"
        for i, a in enumerate(candidates, 1)
    )
    for attempt in range(1, GROQ_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": CLASSIFIER_SYSTEM},
                           {"role": "user",   "content": numbered}],
                temperature=0.0, max_tokens=200,
            )
            raw   = res.choices[0].message.content.strip()
            data  = json.loads(raw)
            idxs  = [i-1 for i in data["selected"] if 1 <= i <= len(candidates)]
            result = [candidates[i] for i in idxs[:TARGET_ARTICLES]]
            log.info(f"Classifier selected {len(result)} articles.")
            return result
        except Exception as e:
            log.warning(f"Classifier attempt {attempt} failed: {e}")
            if attempt < GROQ_RETRY: time.sleep(GROQ_DELAY)

    log.warning("Classifier failed — using top keyword-scored articles.")
    return candidates[:TARGET_ARTICLES]


# ══════════════════════════════════════════════════════════
# SUMMARY  —  Perplexity-style
# ══════════════════════════════════════════════════════════
SUMMARY_SYSTEM = """
You are an AI tech analyst writing for senior engineers and researchers.
Write like Perplexity AI — dense, factual, source-cited, no fluff.

Format your response EXACTLY like this:

## Today's Signal

**[One sharp sentence: the dominant theme in today's tech news]**

---

## Top Stories

### 1. [Article Title]
**Category:** AI Model | Hardware | Cybersecurity | Robotics | Quantum | Regulation | Tools
**Source:** [domain name only]

[2–3 sentence factual summary. Include model names, benchmark numbers, parameter counts, version numbers, chip specs wherever available. No marketing language.]

**Why it matters:** [1 sentence — engineering significance only]

---

[repeat ### N. block for each article]

---

## Key Takeaways
- [Bullet 1: most important technical development]
- [Bullet 2: second most important]
- [Bullet 3: trend or pattern across today's stories]
- [Bullet 4: what engineers/builders should do or watch]

## Signal Strength
**Verdict:** 🟢 Strong Signal | 🟡 Mixed | 🔴 Mostly Noise
**Reason:** [1 sentence]

---
*{n} stories · {date}*
"""

def generate_summary(articles: list[dict]) -> str:
    n = len(articles)
    news_block = "\n\n".join(
        f"{i}. TITLE: {a.get('title','N/A')}\n"
        f"   DESC: {(a.get('description') or 'N/A')[:300]}\n"
        f"   SOURCE: {a.get('source_url') or a.get('link','N/A')}"
        for i, a in enumerate(articles, 1)
    )
    system = SUMMARY_SYSTEM.replace("{n}", str(n)).replace(
        "{date}", datetime.now().strftime("%b %d, %Y"))

    for attempt in range(1, GROQ_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",   # better quality for summary
                messages=[{"role": "system", "content": system},
                           {"role": "user",   "content": f"Analyse these {n} articles:\n\n{news_block}"}],
                temperature=0.3, max_tokens=3500,
            )
            return res.choices[0].message.content
        except Exception as e:
            log.warning(f"Summary attempt {attempt} failed: {e}")
            # fallback to faster model
            try:
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "system", "content": system},
                               {"role": "user", "content": f"Analyse these {n} articles:\n\n{news_block}"}],
                    temperature=0.3, max_tokens=3000,
                )
                return res.choices[0].message.content
            except: pass
            if attempt < GROQ_RETRY: time.sleep(GROQ_DELAY)
    return "[ERROR] Summary generation failed."


# ══════════════════════════════════════════════════════════
# CHATBOT  (called by /api/chat endpoint)
# ══════════════════════════════════════════════════════════
CHAT_SYSTEM = """
You are an AI tech news assistant with deep knowledge of AI, hardware, robotics,
and cybersecurity. You answer questions about technology concisely and accurately.
If the user asks about today's news, use the context provided.
Be direct — no filler phrases like "Great question!" or "Certainly!".
Keep responses under 200 words unless the user asks for more detail.
"""

def chat_response(messages: list[dict], news_context: str = "") -> str:
    system = CHAT_SYSTEM
    if news_context:
        system += f"\n\nToday's news context:\n{news_context[:2000]}"

    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.5, max_tokens=600,
        )
        return res.choices[0].message.content
    except Exception as e:
        log.error(f"Chat error: {e}")
        return "Sorry, I'm having trouble responding right now. Try again in a moment."


# ══════════════════════════════════════════════════════════
# SMART PUSH  (only when articles are actually new)
# ══════════════════════════════════════════════════════════
def _load_prev_hashes() -> set:
    if PREV_HASHES_FILE.exists():
        try: return set(json.loads(PREV_HASHES_FILE.read_text()))
        except: return set()
    return set()

def _save_prev_hashes(titles: list[str]):
    hashes = [hashlib.sha1(t.strip().lower().encode()).hexdigest()[:16] for t in titles]
    PREV_HASHES_FILE.write_text(json.dumps(hashes))

def _count_new(titles: list[str], prev: set) -> int:
    return sum(1 for t in titles
               if hashlib.sha1(t.strip().lower().encode()).hexdigest()[:16] not in prev)

def send_push_notification(article_count: int, titles: list[str]):
    prev_hashes = _load_prev_hashes()
    new_count   = _count_new(titles, prev_hashes)
    _save_prev_hashes(titles)

    if new_count == 0:
        log.info("No new articles since last push — skipping FCM.")
        return

    log.info(f"Sending FCM push for {new_count} new articles.")

    # Try FCM v1 API with service account
    if Path(SERVICE_ACCOUNT_PATH).exists() and FIREBASE_PROJECT_ID:
        try:
            from google.oauth2 import service_account
            from google.auth.transport.requests import Request
            creds = service_account.Credentials.from_service_account_file(
                SERVICE_ACCOUNT_PATH,
                scopes=["https://www.googleapis.com/auth/firebase.messaging"]
            )
            creds.refresh(Request())
            r = requests.post(
                f"https://fcm.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/messages:send",
                json={"message": {
                    "topic": "tech_news",
                    "notification": {
                        "title": f"📡 {new_count} New Tech Articles",
                        "body": titles[0][:80] if titles else "Tap to read your digest"
                    },
                    "data": {"new_count": str(new_count)}
                }},
                headers={"Authorization": f"Bearer {creds.token}",
                         "Content-Type": "application/json"},
                timeout=10,
            )
            log.info(f"FCM v1 response: {r.status_code} {r.text[:200]}")
            return
        except Exception as e:
            log.warning(f"FCM v1 failed: {e} — trying legacy")

    # Fallback: legacy FCM
    if FCM_SERVER_KEY:
        try:
            r = requests.post(
                "https://fcm.googleapis.com/fcm/send",
                json={"to": "/topics/tech_news",
                      "notification": {
                          "title": f"📡 {new_count} New Tech Articles",
                          "body": titles[0][:80] if titles else "Tap to read"
                      }},
                headers={"Authorization": f"key={FCM_SERVER_KEY}",
                         "Content-Type": "application/json"},
                timeout=10,
            )
            log.info(f"FCM legacy response: {r.status_code}")
        except Exception as e:
            log.error(f"FCM legacy failed: {e}")


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline() -> dict:
    memory = ArticleMemory()
    log.info(f"Pipeline started. Memory: {memory.size()} seen articles.")

    raw = fetch_news()
    if not raw:
        return {"status": "error", "message": "News API returned no results.",
                "articles": [], "summary": "", "run_at": datetime.now().isoformat()}

    candidates = prefilter(raw, memory)

    # If still too few, lower the bar and retry with all raw
    if len(candidates) < 4:
        log.warning(f"Only {len(candidates)} candidates — relaxing memory filter")
        memory = ArticleMemory()
        candidates = prefilter(raw, memory)
        # init empty memory object
        empty_mem = object.__new__(ArticleMemory)
        empty_mem._data = {}
        empty_mem.path = MEMORY_FILE
        candidates = prefilter(raw, empty_mem)

    if not candidates:
        return {"status": "error", "message": "No relevant articles found.",
                "articles": [], "summary": "", "run_at": datetime.now().isoformat()}

    selected = llm_classify(candidates)
    if not selected:
        selected = candidates[:TARGET_ARTICLES]

    summary = generate_summary(selected)
    memory.mark_batch([a.get("title", "") for a in selected])

    titles = [a.get("title", "") for a in selected]
    send_push_notification(len(selected), titles)

    log.info(f"Pipeline complete. {len(selected)} articles.")
    return {
        "status":        "ok",
        "run_at":        datetime.now().isoformat(),
        "article_count": len(selected),
        "articles": [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "source_url":  a.get("source_url") or a.get("link", ""),
                "image_url":   a.get("image_url", ""),
                "pubDate":     a.get("pubDate", ""),
            }
            for a in selected
        ],
        "summary": summary,
    }