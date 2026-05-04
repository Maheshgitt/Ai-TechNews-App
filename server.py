"""
server.py  v4
─────────────
GET  /api/news        → cached digest (instant)
POST /api/refresh     → manual pipeline trigger
POST /api/chat        → Groq chatbot
GET  /health          → liveness
"""

import json, logging, os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline import run_pipeline, chat_response

load_dotenv()
log = logging.getLogger("server")

DAILY_RUN_TIME = os.getenv("DAILY_RUN_TIME", "08:00")
CACHE_FILE = Path(__file__).parent / "cache" / "history.json"
CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── cache ──────────────────────────────────────────────────
def save_cache(new_data: dict):
    try:
        history = []

        if CACHE_FILE.exists():
            history = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        # Add timestamp
        new_data["saved_at"] = datetime.utcnow().isoformat()

        # Append new run
        history.append(new_data)

        # Keep only last 24 hours
        now = datetime.utcnow()
        filtered = []

        for item in history:
            try:
                t = datetime.fromisoformat(item["saved_at"])
                if (now - t).total_seconds() <= 86400:
                    filtered.append(item)
            except:
                continue

        CACHE_FILE.write_text(
            json.dumps(filtered, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        log.info(f"History saved. Entries: {len(filtered)}")

    except Exception as e:
        log.error(f"Cache save failed: {e}")

def load_cache():
    if not CACHE_FILE.exists():
        return []

    try:
        history = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

        now = datetime.utcnow()
        valid = []

        for item in history:
            try:
                t = datetime.fromisoformat(item["saved_at"])
                if (now - t).total_seconds() <= 86400:
                    valid.append(item)
            except:
                continue

        return valid

    except:
        return []


# ── scheduler ─────────────────────────────────────────────
def scheduled_run():
    log.info("Scheduled pipeline run starting.")
    result = run_pipeline()
    save_cache(result)
    log.info(f"Scheduled run done. Status: {result['status']}")


# ── app lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    hour, minute = map(int, DAILY_RUN_TIME.split(":"))
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(scheduled_run, "cron", hour=hour, minute=minute)

    # Also run every 6 hours to catch breaking news
    scheduler.add_job(scheduled_run, "interval", hours=6, id="interval_run")

    scheduler.start()
    log.info(f"Scheduler started — daily at {DAILY_RUN_TIME} UTC + every 6 hrs.")
    yield
    scheduler.shutdown()


app = FastAPI(title="AI Tech News API", version="4.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── models ────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    include_news_context: bool = True


# ── routes ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/api/news")
def get_news():
    """
    Returns cached digest immediately.
    If no cache exists, runs pipeline synchronously (first boot only).
    """
@app.get("/api/news")
def get_news():
    history = load_cache()

    # If history exists → return all
    if history:
        return history

    # First time → generate
    log.info("No history — running pipeline")
    result = run_pipeline()
    save_cache(result)

    return [result]


@app.post("/api/refresh")
async def refresh_news(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_and_cache)
    return {"status": "accepted",
            "message": "Pipeline started. Poll /api/news in 30 seconds."}


async def _run_and_cache():
    result = run_pipeline()
    save_cache(result)
    log.info(f"Manual refresh done. Articles: {result.get('article_count', 0)}")


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    Groq-powered chatbot endpoint.
    Optionally injects today's news as context.
    """
    news_context = ""
    if req.include_news_context:
        cached = load_cache()
        if cached and cached.get("summary"):
            news_context = cached["summary"]

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    reply = chat_response(messages, news_context)
    return {"reply": reply, "model": "llama-3.3-70b-versatile"}