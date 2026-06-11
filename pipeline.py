"""
pipeline.py  v7
───────────────
Removed: image_gen.py, enrich_with_images(), ai_image_url
Added:   image_url passed straight from NewsData.io response
"""

import re, json, time, hashlib, logging, os, requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

NEWS_API_KEY = os.getenv("NEWSDATA_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
if not NEWS_API_KEY: raise ValueError("NEWSDATA_API_KEY missing")
if not GROQ_API_KEY: raise ValueError("GROQ_API_KEY missing")

BASE_DIR    = Path(__file__).parent
MEMORY_FILE = BASE_DIR / "memory" / "seen_articles.json"
LOG_DIR     = BASE_DIR / "logs"
MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            LOG_DIR / f"run_{datetime.now():%Y-%m-%d}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("pipeline")

TARGET_ARTICLES    = 8
MIN_SCORE          = 3
DEDUP_RATIO        = 0.72
MEMORY_EXPIRY_DAYS = 1
GROQ_RETRY         = 3
GROQ_DELAY         = 2

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
# MEMORY
# ══════════════════════════════════════════════════════════
class ArticleMemory:
    def __init__(self, path: Path = MEMORY_FILE, empty: bool = False):
        self.path  = path
        self._data: dict[str, str] = {}
        if not empty:
            self._load()
            self._purge()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception:
                self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def _purge(self):
        cutoff = datetime.now() - timedelta(days=MEMORY_EXPIRY_DAYS)
        self._data = {
            h: ts for h, ts in self._data.items()
            if datetime.fromisoformat(ts) > cutoff
        }
        self._save()

    @staticmethod
    def _hash(title: str) -> str:
        return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]

    def seen(self, title: str) -> bool:
        return self._hash(title) in self._data

    def mark_batch(self, titles: list[str]):
        now = datetime.now().isoformat()
        for t in titles:
            self._data[self._hash(t)] = now
        self._save()

    def size(self) -> int:
        return len(self._data)


# ══════════════════════════════════════════════════════════
# FETCH
# ══════════════════════════════════════════════════════════
def fetch_news() -> list[dict]:
    all_articles = []
    for cat in ["technology", "science"]:
        try:
            r = requests.get(
                "https://newsdata.io/api/1/news",
                params={
                    "apikey":   NEWS_API_KEY,
                    "category": cat,
                    "language": "en",
                    "size":     10,
                },
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
    return sum(
        w for kw, w in KEYWORD_WEIGHTS.items()
        if re.search(r"\b" + re.escape(kw) + r"\b", text)
    )

def _src_score(article: dict) -> int:
    url = (article.get("source_url") or article.get("link") or "").lower()
    for domain, bonus in SOURCE_SCORES.items():
        if domain in url:
            return bonus
    return 0

def _is_dup(new_title: str, seen: list[str]) -> bool:
    return any(
        SequenceMatcher(None, new_title.lower(), s.lower()).ratio() >= DEDUP_RATIO
        for s in seen
    )

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
        if total >= MIN_SCORE:
            seen_titles.append(title)
            scored.append((total, a))
    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = [a for _, a in scored[:25]]
    log.info(f"Pre-filter: {len(candidates)} candidates.")
    return candidates


# ══════════════════════════════════════════════════════════
# LLM CLASSIFIER
# ══════════════════════════════════════════════════════════
CLASSIFIER_SYSTEM = f"""
You are a strict tech news classifier. Select the top {TARGET_ARTICLES} most
impactful articles. Prioritise: AI models, hardware, robotics, cybersecurity,
open-source. Reject: finance, celebrity, generic PR.
Return ONLY valid JSON: {{"selected": [1, 2, 5, ...]}}
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
                messages=[
                    {"role": "system", "content": CLASSIFIER_SYSTEM},
                    {"role": "user",   "content": numbered},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw    = res.choices[0].message.content.strip()
            data   = json.loads(raw)
            idxs   = [i - 1 for i in data["selected"] if 1 <= i <= len(candidates)]
            result = [candidates[i] for i in idxs[:TARGET_ARTICLES]]
            log.info(f"Classifier selected {len(result)} articles.")
            return result
        except Exception as e:
            log.warning(f"Classifier attempt {attempt} failed: {e}")
            if attempt < GROQ_RETRY:
                time.sleep(GROQ_DELAY)
    return candidates[:TARGET_ARTICLES]


# ══════════════════════════════════════════════════════════
# PER-ARTICLE SUMMARY
# ══════════════════════════════════════════════════════════
ARTICLE_SUMMARY_SYSTEM = """
Write a 3-sentence factual technical summary for engineers.
Include: what happened, key specs/numbers if available, significance.
No fluff. Return ONLY the summary text.
"""

def _summarise_one(article: dict) -> str:
    title = article.get("title", "")
    desc  = article.get("description", "") or ""
    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": ARTICLE_SUMMARY_SYSTEM},
                {"role": "user",   "content": f"Title: {title}\nDescription: {desc[:400]}"},
            ],
            temperature=0.2,
            max_tokens=200,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"Article summary failed for '{title[:40]}': {e}")
        return desc[:300]


# ══════════════════════════════════════════════════════════
# GLOBAL SUMMARY
# ══════════════════════════════════════════════════════════
SUMMARY_SYSTEM = """
You are an AI tech analyst writing for senior engineers. Write like Perplexity AI.

## Today's Signal
**[One sharp sentence: dominant theme]**

---

## Top Stories

### N. [Article Title]
**Category:** AI Model | Hardware | Cybersecurity | Robotics | Quantum | Regulation | Tools
**Source:** [domain only]

[2–3 sentence factual summary with specs, benchmark numbers where available.]

**Why it matters:** [1 sentence engineering significance]

---

## Key Takeaways
- [Most important technical development]
- [Second most important]
- [Cross-story trend]
- [What engineers should watch]

## Signal Strength
**Verdict:** 🟢 Strong Signal | 🟡 Mixed | 🔴 Mostly Noise
**Reason:** [1 sentence]

---
*{n} stories · {date}*
"""

def generate_global_summary(articles: list[dict]) -> str:
    n = len(articles)
    news_block = "\n\n".join(
        f"{i}. TITLE: {a.get('title', 'N/A')}\n"
        f"   DESC: {(a.get('description') or 'N/A')[:300]}\n"
        f"   SOURCE: {a.get('source_url') or a.get('link', 'N/A')}"
        for i, a in enumerate(articles, 1)
    )
    system = SUMMARY_SYSTEM.replace("{n}", str(n)).replace(
        "{date}", datetime.now().strftime("%b %d, %Y")
    )
    for attempt in range(1, GROQ_RETRY + 1):
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": f"Analyse {n} articles:\n\n{news_block}"},
                ],
                temperature=0.3,
                max_tokens=3500,
            )
            return res.choices[0].message.content
        except Exception as e:
            log.warning(f"Global summary attempt {attempt} failed: {e}")
            try:
                res = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": f"Analyse {n} articles:\n\n{news_block}"},
                    ],
                    temperature=0.3,
                    max_tokens=3000,
                )
                return res.choices[0].message.content
            except Exception:
                pass
            if attempt < GROQ_RETRY:
                time.sleep(GROQ_DELAY)
    return "[ERROR] Summary generation failed."


# ══════════════════════════════════════════════════════════
# CHATBOT
# ══════════════════════════════════════════════════════════
CHAT_SYSTEM = """
You are an AI tech news assistant. Answer questions about technology concisely.
Use today's news context when relevant. Be direct. Under 200 words unless asked for more.
"""

def chat_response(messages: list[dict], news_context: str = "") -> str:
    system = CHAT_SYSTEM
    if news_context:
        system += f"\n\nToday's full news context:\n{news_context[:4000]}"
    try:
        res = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": system}] + messages,
            temperature=0.5,
            max_tokens=600,
        )
        return res.choices[0].message.content
    except Exception as e:
        log.error(f"Chat error: {e}")
        return "Sorry, having trouble responding. Try again in a moment."


# ══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════
def run_pipeline() -> dict:
    memory = ArticleMemory()
    log.info(f"Pipeline v7 started. Memory: {memory.size()} seen articles.")

    raw = fetch_news()
    if not raw:
        return {
            "status": "error", "message": "News API returned no results.",
            "articles": [], "summary": "", "run_at": datetime.now().isoformat(),
        }

    candidates = prefilter(raw, memory)
    if len(candidates) < 4:
        log.warning("Too few candidates — bypassing memory filter.")
        candidates = prefilter(raw, ArticleMemory(empty=True))

    if not candidates:
        return {
            "status": "error", "message": "No relevant articles found.",
            "articles": [], "summary": "", "run_at": datetime.now().isoformat(),
        }

    selected = llm_classify(candidates)
    if not selected:
        selected = candidates[:TARGET_ARTICLES]

    # Per-article summaries + global digest in parallel
    log.info("Generating summaries in parallel…")
    with ThreadPoolExecutor(max_workers=4) as executor:
        summary_futures = {
            executor.submit(_summarise_one, a): i
            for i, a in enumerate(selected)
        }
        global_future = executor.submit(generate_global_summary, selected)

        for future, idx in summary_futures.items():
            try:
                selected[idx]["ai_summary"] = future.result()
            except Exception:
                selected[idx]["ai_summary"] = selected[idx].get("description", "")

        global_summary = global_future.result()

    memory.mark_batch([a.get("title", "") for a in selected])
    log.info(f"Pipeline v7 complete. {len(selected)} articles.")

    return {
        "status":        "ok",
        "run_at":        datetime.now().isoformat(),
        "article_count": len(selected),
        "articles": [
            {
                "title":       a.get("title", ""),
                "description": a.get("description", ""),
                "summary":     a.get("ai_summary", ""),
                # image_url comes directly from NewsData.io — empty string if not provided
                "image_url":   a.get("image_url") or "",
                "source_url":  a.get("source_url") or a.get("link", ""),
                "pubDate":     a.get("pubDate", ""),
            }
            for a in selected
        ],
        "summary": global_summary,
    }