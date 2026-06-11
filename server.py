"""
server.py  v6
──────────────────────────────────────────────────
Features:
- Latest news API
- History API
- AI Chat API
- Firebase Cloud Messaging notifications
- Scheduled refresh
- 24-hour history retention

Removed:
- Firebase Storage
- AI image hosting
- Pollinations image generation
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

DAILY_RUN_TIME    = os.getenv("DAILY_RUN_TIME", "08:00")
POLL_INTERVAL_MIN = int(os.getenv("POLL_INTERVAL_MIN", "10"))
BASE_DIR          = Path(__file__).parent
HISTORY_FILE      = BASE_DIR / "cache" / "history.json"
PUSH_HASHES_FILE  = BASE_DIR / "cache" / "push_hashes.json"
HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Firebase init (FCM notifications) ────────────────────
_firebase_ready = False

def _init_firebase():
    global _firebase_ready

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not sa_path or not Path(sa_path).exists():
        log.warning("GOOGLE_APPLICATION_CREDENTIALS missing — FCM disabled.")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials

        if not firebase_admin._apps:
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)

        _firebase_ready = True
        log.info("Firebase initialized successfully (FCM enabled).")

    except Exception as e:
        log.error(f"Firebase init failed: {e}")


# ══════════════════════════════════════════════════════════
# HISTORY CACHE
# ══════════════════════════════════════════════════════════
def _parse_ts(ts: str | None) -> datetime:
    try:
        return datetime.fromisoformat(ts) if ts else datetime.min
    except Exception:
        return datetime.min

def save_to_history(data: dict):
    history = _load_history_raw()
    data["saved_at"] = datetime.utcnow().isoformat()
    history.append(data)
    cutoff  = datetime.utcnow() - timedelta(hours=24)
    history = [h for h in history if _parse_ts(h.get("saved_at")) > cutoff]
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def _load_history_raw() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []

def get_history() -> list[dict]:
    cutoff  = datetime.utcnow() - timedelta(hours=24)
    history = _load_history_raw()
    valid   = [h for h in history if _parse_ts(h.get("saved_at")) > cutoff]
    return list(reversed(valid))

def get_latest() -> dict | None:
    history = get_history()
    return history[0] if history else None


# ══════════════════════════════════════════════════════════
# FCM PUSH
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
    if not _firebase_ready or not articles:
        return
    prev  = _load_push_hashes()
    new_a = [a for a in articles if _title_hash(a.get("title", "")) not in prev]
    if not new_a:
        log.info("Push skipped — no new articles.")
        return
    all_hashes = prev | {_title_hash(a.get("title", "")) for a in articles}
    _save_push_hashes(all_hashes)
    try:
        from firebase_admin import messaging
        messaging.send(messaging.Message(
            notification=messaging.Notification(
                title=f"📡 {len(new_a)} New Tech {'Article' if len(new_a)==1 else 'Articles'}",
                body=new_a[0].get("title", "Tap to read")[:80],
            ),
            data={"article_count": str(len(new_a))},
            topic="tech_news",
        ))
        log.info(f"FCM push sent for {len(new_a)} new articles.")
    except Exception as e:
        log.error(f"FCM push failed: {e}")


# ══════════════════════════════════════════════════════════
# PIPELINE RUNNER
# ══════════════════════════════════════════════════════════
def run_and_store():
    result = run_pipeline()
    if result.get("status") == "ok" and result.get("articles"):
        save_to_history(result)
        send_push_if_new(result["articles"])
        log.info(f"Stored {result['article_count']} articles.")
    else:
        log.warning(f"Pipeline no articles: {result.get('message','')}")
    return result


# ══════════════════════════════════════════════════════════
# LIFESPAN
# ══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_firebase()
    hour, minute = map(int, DAILY_RUN_TIME.split(":"))
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(run_and_store, "cron", hour=hour, minute=minute, id="daily")
    scheduler.add_job(run_and_store, "interval", minutes=POLL_INTERVAL_MIN, id="poll")
    scheduler.start()
    log.info(f"Scheduler: daily at {DAILY_RUN_TIME} UTC + every {POLL_INTERVAL_MIN}min")
    if not get_latest():
        import threading
        threading.Thread(target=run_and_store, daemon=True).start()
    yield
    scheduler.shutdown()

app = FastAPI(title="AI Tech News API", version="6.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ══════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════
class ChatMessage(BaseModel):
    role: str
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
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "history_count": len(get_history()),
        "latest_run": latest.get("run_at") if latest else None,
        "firebase": _firebase_ready,
    }

@app.get("/api/latest")
def api_latest():
    latest = get_latest()
    if latest:
        return latest
    result = run_pipeline()
    if result.get("status") == "ok":
        save_to_history(result)
    return result

@app.get("/api/history")
def api_history():
    return get_history()

@app.post("/api/refresh")
async def refresh_news(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_and_store)
    return {"status": "accepted", "message": "Pipeline started. Check /api/latest in ~60s."}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    news_context = ""
    if req.include_news_context:
        history = get_history()
        if history:
            parts = []
            for i, batch in enumerate(reversed(history)):
                ts      = batch.get("saved_at", "")[:16].replace("T", " ")
                summary = batch.get("summary", "").strip()
                if summary:
                    parts.append(f"[Batch {i+1} · {ts} UTC]\n{summary}")
            news_context = "\n\n---\n\n".join(parts)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply    = chat_response(messages, news_context)
    return {"reply": reply, "model": "llama-3.3-70b-versatile"}