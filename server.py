"""
server.py  v5
─────────────
Fixes applied:
  ✅ UnboundLocalError in _run_and_cache (latest_article scope)
  ✅ Indentation bug in /api/chat (news_context always ran)
  ✅ firebase_admin double-init crash (try/except guard)
  ✅ GOOGLE_APPLICATION_CREDENTIALS=None crash (existence check)
  ✅ Duplicate push notifications (removed from pipeline, centralised here)
  ✅ /api/news returns DigestResponse (single), /api/history returns list

New endpoints:
  GET  /api/latest    → most recent single digest (Android homepage)
  GET  /api/history   → all batches today (Android history tab)
  POST /api/refresh   → manual trigger
  POST /api/chat      → Groq chatbot with full-day context
  GET  /health        → liveness

Architecture:
  • Real-time poller runs every 10 minutes, checks for NEW articles only
  • FCM push sent only when new unseen articles found
  • History file stores all today's batches, auto-purges after 24 hrs
"""

import json, logging, os, hashlib
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import run_pipeline, chat_response

load_dotenv()

log = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)

# ── config ─────────────────────────────────────────────────
DAILY_RUN_TIME   = os.getenv("DAILY_RUN_TIME", "08:00")
POLL_INTERVAL_MIN = int(os.getenv("POLL_INTERVAL_MIN", "10"))  # check for new articles every N min
BASE_DIR         = Path(__file__).parent
HISTORY_FILE     = BASE_DIR / "cache" / "history.json"
PUSH_HASHES_FILE = BASE_DIR / "cache" / "push_hashes.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Firebase init (guarded) ────────────────────────────────
_firebase_ready = False

def _init_firebase():
    global _firebase_ready
    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not sa_path or not Path(sa_path).exists():
        log.warning("GOOGLE_APPLICATION_CREDENTIALS not set or file missing — FCM disabled.")
        return
    try:
        import firebase_admin
        from firebase_admin import credentials
        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)
        _firebase_ready = True
        log.info("Firebase initialized successfully.")
    except Exception as e:
        log.error(f"Firebase init failed: {e}")


# ══════════════════════════════════════════════════════════
# HISTORY CACHE  (list of batches, 24-hr TTL)
# ══════════════════════════════════════════════════════════
def save_to_history(data: dict):
    """Append a pipeline result to today's history. Purge entries > 24 hrs."""
    history = _load_history_raw()
    data["saved_at"] = datetime.utcnow().isoformat()
    history.append(data)
    # Keep only last 24 hrs
    cutoff  = datetime.utcnow() - timedelta(hours=24)
    history = [
        item for item in history
        if _parse_ts(item.get("saved_at")) > cutoff
    ]
    HISTORY_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def _parse_ts(ts_str: str | None) -> datetime:
    try:
        return datetime.fromisoformat(ts_str) if ts_str else datetime.min
    except Exception:
        return datetime.min

def _load_history_raw() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def get_history() -> list[dict]:
    """Return all batches from the last 24 hrs, newest first."""
    cutoff  = datetime.utcnow() - timedelta(hours=24)
    history = _load_history_raw()
    valid   = [h for h in history if _parse_ts(h.get("saved_at")) > cutoff]
    return list(reversed(valid))   # newest first

def get_latest() -> dict | None:
    """Return the single most-recent batch."""
    history = get_history()
    return history[0] if history else None


# ══════════════════════════════════════════════════════════
# FCM PUSH  (only for genuinely new articles)
# ══════════════════════════════════════════════════════════
def _load_push_hashes() -> set:
    if PUSH_HASHES_FILE.exists():
        try:
            return set(json.loads(PUSH_HASHES_FILE.read_text()))
        except Exception:
            return set()
    return set()

def _save_push_hashes(hashes: set):
    PUSH_HASHES_FILE.write_text(json.dumps(list(hashes)))

def _title_hash(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]

def send_push_if_new(articles: list[dict]):
    """
    Send FCM push notification only if any article is new since last push.
    Deduplication is done via persistent hash file.
    """
    if not _firebase_ready or not articles:
        return

    prev_hashes = _load_push_hashes()
    new_articles = [
        a for a in articles
        if _title_hash(a.get("title", "")) not in prev_hashes
    ]

    if not new_articles:
        log.info("Push skipped — no new articles since last notification.")
        return

    # Update hash store BEFORE sending (prevents retry loops)
    all_hashes = prev_hashes | {_title_hash(a.get("title", "")) for a in articles}
    _save_push_hashes(all_hashes)

    count      = len(new_articles)
    top_title  = new_articles[0].get("title", "New tech articles available")
    top_desc   = new_articles[0].get("description", "")
    notif_body = top_desc[:100] if top_desc else f"{count} new articles available"

    try:
        from firebase_admin import messaging
        message = messaging.Message(
            notification=messaging.Notification(
                title=f"📡 {count} New Tech {'Article' if count == 1 else 'Articles'}",
                body=top_title[:80],
            ),
            data={"article_count": str(count), "top_title": top_title[:80]},
            topic="tech_news",   # all subscribed devices receive this
        )
        response = messaging.send(message)
        log.info(f"FCM push sent for {count} new articles. Response: {response}")
    except Exception as e:
        log.error(f"FCM push failed: {e}")


# ══════════════════════════════════════════════════════════
# PIPELINE RUNNER  (used by scheduler + manual refresh)
# ══════════════════════════════════════════════════════════
def run_and_store():
    """Run the news pipeline, store result in history, send push if new."""
    log.info("Running pipeline...")
    result = run_pipeline()

    if result.get("status") == "ok" and result.get("articles"):
        save_to_history(result)
        send_push_if_new(result["articles"])
        log.info(f"Stored {result['article_count']} articles in history.")
    else:
        log.warning(f"Pipeline returned no articles: {result.get('message','')}")

    return result


# ══════════════════════════════════════════════════════════
# APP LIFESPAN + SCHEDULER
# ══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_firebase()

    hour, minute = map(int, DAILY_RUN_TIME.split(":"))
    scheduler = BackgroundScheduler(timezone="UTC")

    # Full daily run
    scheduler.add_job(run_and_store, "cron", hour=hour, minute=minute,
                      id="daily_run")

    # Real-time polling — checks for new articles every POLL_INTERVAL_MIN
    scheduler.add_job(run_and_store, "interval",
                      minutes=POLL_INTERVAL_MIN,
                      id="realtime_poll")

    scheduler.start()
    log.info(
        f"Scheduler started — daily at {DAILY_RUN_TIME} UTC "
        f"+ polling every {POLL_INTERVAL_MIN} min."
    )

    # Run once immediately on startup if no history exists
    if not get_latest():
        log.info("No cached history — running initial pipeline.")
        import threading
        threading.Thread(target=run_and_store, daemon=True).start()

    yield
    scheduler.shutdown()


app = FastAPI(title="AI Tech News API", version="5.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str      # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    include_news_context: bool = True


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/health")
def health():
    latest = get_latest()
    return {
        "status":       "ok",
        "time":         datetime.utcnow().isoformat(),
        "history_count": len(get_history()),
        "latest_run":   latest.get("run_at") if latest else None,
        "firebase":     _firebase_ready,
    }


@app.get("/api/latest")
def api_latest():
    """
    Returns the most recent news batch as a single DigestResponse object.
    Android homepage calls this — expects one dict, not a list.
    If no cache exists, runs pipeline synchronously (first boot).
    """
    latest = get_latest()
    if latest:
        return latest

    # First boot — run synchronously so the app doesn't get 503
    log.info("No history on /api/latest — running pipeline now.")
    result = run_pipeline()
    if result.get("status") == "ok":
        save_to_history(result)
    return result


@app.get("/api/history")
def api_history():
    """
    Returns all news batches from the last 24 hrs (newest first).
    Android history tab calls this.
    Each item has: run_at, saved_at, article_count, articles[], summary
    """
    history = get_history()
    if not history:
        return []
    return history


@app.post("/api/refresh")
async def refresh_news(background_tasks: BackgroundTasks):
    """Trigger an immediate pipeline run in the background."""
    background_tasks.add_task(run_and_store)
    return {
        "status":  "accepted",
        "message": "Pipeline started. Check /api/latest in ~30 seconds.",
    }


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Groq chatbot with FULL daily news context (all batches, not just latest).
    """
    news_context = ""

    if req.include_news_context:
        history = get_history()
        if history:
            # Build context from ALL today's summaries so chatbot knows everything
            parts = []
            for i, batch in enumerate(reversed(history)):   # chronological order
                ts      = batch.get("saved_at", "")[:16].replace("T", " ")
                summary = batch.get("summary", "").strip()
                if summary:
                    parts.append(f"[Batch {i+1} · {ts} UTC]\n{summary}")
            news_context = "\n\n---\n\n".join(parts)

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply    = chat_response(messages, news_context)

    return {"reply": reply, "model": "llama-3.3-70b-versatile"}