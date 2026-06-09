"""
image_gen.py  —  AI image generation module
─────────────────────────────────────────────
Provider: Pollinations AI (free, no API key, URL-based)
Fallback: keyword template (instant, zero cost)

Designed for easy future migration to:
  Hugging Face Inference API  →  swap _call_provider()
  Flux / SDXL via RunPod     →  swap _call_provider()
  DALL-E / Ideogram           →  swap _call_provider()

Public API:
  enrich_with_images(articles, groq_client) → list[dict]
    Adds "ai_image_url" and "ai_prompt" to each article dict.
    Parallel execution, cache-aware, deduplication-safe.
"""

import json
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

log = logging.getLogger("image_gen")

# ── paths ─────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
IMAGE_CACHE_FILE = BASE_DIR / "cache" / "image_cache.json"
IMAGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Pollinations config ───────────────────────────────────
POLLINATIONS_BASE = "https://image.pollinations.ai/prompt"
IMG_WIDTH         = 800
IMG_HEIGHT        = 450
IMG_MODEL         = "flux"          # options: flux, flux-realism, flux-anime, turbo

# ── max parallel workers for prompt generation ────────────
MAX_WORKERS = 4

# ──────────────────────────────────────────────────────────
# KEYWORD TEMPLATE LIBRARY
# Fast, zero-cost fallback. Covers ~90% of tech news topics.
# ──────────────────────────────────────────────────────────
PROMPT_TEMPLATES: list[tuple[list[str], str]] = [
    (
        ["openai", "gpt", "chatgpt", "o3", "o4", "sora", "dall-e"],
        "Advanced AI research laboratory with neural networks visualised as glowing blue connections, "
        "scientists working at holographic terminals, futuristic environment, cinematic lighting, ultra detailed"
    ),
    (
        ["anthropic", "claude"],
        "Futuristic AI safety research facility, glowing circuits forming a brain-like structure, "
        "clean white laboratory with blue accents, researchers studying AI alignment, cinematic lighting"
    ),
    (
        ["gemini", "deepmind", "google ai", "google deepmind"],
        "Google AI research lab with colourful holographic data visualisations, "
        "quantum computing equipment, advanced neural network displays, cinematic lighting, ultra detailed"
    ),
    (
        ["llama", "meta ai", "mistral", "deepseek", "qwen", "open-source"],
        "Open source AI development hub with collaborative screens showing code and model training graphs, "
        "diverse team of engineers, modern tech workspace, cinematic lighting"
    ),
    (
        ["nvidia", "gpu", "h100", "b200", "cuda"],
        "Futuristic AI datacenter with rows of NVIDIA GPUs emitting green light, "
        "advanced server racks with glowing components, modern technology visualisation, cinematic lighting"
    ),
    (
        ["chip", "semiconductor", "processor", "tsmc", "intel", "amd", "npu", "tpu", "asic"],
        "Extreme close-up of a semiconductor chip with intricate circuit patterns, "
        "silicon wafer manufacturing in clean room, microscopic precision engineering, cinematic macro photography"
    ),
    (
        ["quantum computing", "quantum chip", "qubit"],
        "Quantum computer with glowing superconducting circuits suspended in cryogenic chamber, "
        "liquid helium cooling system, complex quantum circuitry, cinematic blue lighting, ultra detailed"
    ),
    (
        ["robot", "robotics", "humanoid", "boston dynamics", "figure ai"],
        "Advanced humanoid robot working in high-tech laboratory, "
        "smooth metallic surfaces, precision mechanical joints, futuristic environment, cinematic lighting, ultra detailed"
    ),
    (
        ["self-driving", "autonomous vehicle", "waymo", "tesla autopilot"],
        "Autonomous vehicle navigating a futuristic smart city at night, "
        "AI sensor arrays glowing, digital HUD overlays, smooth roads with connected infrastructure, cinematic lighting"
    ),
    (
        ["cybersecurity", "zero-day", "ransomware", "cve", "vulnerability", "exploit", "data breach"],
        "Digital cybersecurity shield protecting a glowing network, "
        "binary code streams, firewall visualisation, dark background with electric blue accents, cinematic style"
    ),
    (
        ["5g", "6g", "starlink", "satellite"],
        "Satellite communication network over Earth from orbit, "
        "glowing signal beams connecting cities, futuristic wireless infrastructure, cinematic space photography"
    ),
    (
        ["ai regulation", "ai safety", "ai act", "ai governance"],
        "Government chamber with digital AI ethics framework displayed on large screens, "
        "lawmakers reviewing neural network diagrams, formal environment, cinematic lighting"
    ),
    (
        ["large language model", "llm", "foundation model", "transformer", "neural network", "deep learning"],
        "Abstract visualisation of a large language model, "
        "billions of parameters as glowing nodes in a vast neural network, "
        "flowing data streams, cinematic lighting, ultra detailed"
    ),
    (
        ["apple silicon", "apple"],
        "Apple chip on precision circuit board, clean minimalist design, "
        "micro-engineering precision, silicon photonics, cinematic product photography, ultra detailed"
    ),
    (
        ["startup", "innovation", "breakthrough"],
        "Modern tech startup workspace with engineers collaborating on AI projects, "
        "multiple holographic screens showing code and data, creative futuristic environment, cinematic lighting"
    ),
]

def _keyword_prompt(title: str, description: str) -> str | None:
    """Return a template prompt if any keyword matches. Else None."""
    combined = (title + " " + (description or "")).lower()
    for keywords, prompt in PROMPT_TEMPLATES:
        if any(re.search(r"\b" + re.escape(kw) + r"\b", combined) for kw in keywords):
            return prompt
    return None

# ──────────────────────────────────────────────────────────
# GROQ PROMPT GENERATION  (fallback for non-matched articles)
# ──────────────────────────────────────────────────────────
GROQ_PROMPT_SYSTEM = """
You are a visual art director creating prompts for AI image generation.

Given a tech news article, generate one highly visual, cinematic image prompt.

RULES:
- Describe a SCENE, not an infographic or text
- No text, letters, logos, or brand names in the image
- Futuristic / cinematic tech aesthetic
- End every prompt with: cinematic lighting, ultra detailed
- Maximum 80 words
- Return ONLY the prompt text, nothing else
"""

def _groq_prompt(title: str, description: str, groq_client) -> str:
    """Use Groq to generate a bespoke image prompt."""
    user_msg = f"Article title: {title}\nDescription: {(description or '')[:300]}"
    try:
        res = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": GROQ_PROMPT_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=120,
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        log.warning(f"Groq prompt failed for '{title[:40]}': {e}")
        return (
            "Advanced technology research laboratory with glowing digital interfaces, "
            "scientists working on cutting-edge AI systems, cinematic lighting, ultra detailed"
        )

def generate_prompt(title: str, description: str, groq_client) -> str:
    """
    Hybrid strategy:
      1. Try keyword template first (instant, free)
      2. Fall back to Groq (for unrecognised topics)
    """
    prompt = _keyword_prompt(title, description)
    if prompt:
        log.debug(f"Template prompt used for: {title[:50]}")
        return prompt
    log.debug(f"Groq prompt used for: {title[:50]}")
    return _groq_prompt(title, description, groq_client)

# ──────────────────────────────────────────────────────────
# POLLINATIONS URL BUILDER
# ──────────────────────────────────────────────────────────
def build_image_url(prompt: str, seed: int = 42) -> str:
    """
    Build a Pollinations AI image URL.
    The URL IS the image — Android loads it lazily via Coil.
    No HTTP call needed from backend.

    To migrate to a different provider, replace this function only.
    The rest of the module stays the same.
    """
    encoded = quote(prompt, safe="")
    return (
        f"{POLLINATIONS_BASE}/{encoded}"
        f"?width={IMG_WIDTH}&height={IMG_HEIGHT}"
        f"&nologo=true&model={IMG_MODEL}&seed={seed}"
    )

# ──────────────────────────────────────────────────────────
# IMAGE CACHE  (title-hash → url)
# ──────────────────────────────────────────────────────────
def _title_hash(title: str) -> str:
    return hashlib.sha1(title.strip().lower().encode()).hexdigest()[:16]

def _load_cache() -> dict:
    if IMAGE_CACHE_FILE.exists():
        try:
            return json.loads(IMAGE_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict):
    try:
        IMAGE_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        log.error(f"Image cache save failed: {e}")

# ──────────────────────────────────────────────────────────
# MAIN PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────
def enrich_with_images(articles: list[dict], groq_client) -> list[dict]:
    """
    Add "ai_image_url" and "ai_prompt" to each article dict.

    - Cache-aware: never regenerates for a seen title
    - Parallel: up to MAX_WORKERS concurrent Groq calls
    - Graceful: on failure, sets ai_image_url = "" (Android skips image)
    """
    if not articles:
        return articles

    cache = _load_cache()
    cache_updated = False

    def process(article: dict) -> dict:
        nonlocal cache_updated
        title = article.get("title", "").strip()
        if not title:
            article["ai_image_url"] = ""
            article["ai_prompt"]    = ""
            return article

        h = _title_hash(title)

        # Cache hit — reuse existing URL
        if h in cache:
            log.debug(f"Cache hit for: {title[:50]}")
            article["ai_image_url"] = cache[h]["url"]
            article["ai_prompt"]    = cache[h]["prompt"]
            return article

        # Cache miss — generate prompt + build URL
        prompt = generate_prompt(title, article.get("description", ""), groq_client)
        url    = build_image_url(prompt, seed=int(h[:8], 16) % 10000)

        # Deterministic seed from title hash so same article always gets same image
        cache[h] = {
            "url":        url,
            "prompt":     prompt,
            "created_at": datetime.now().isoformat(),
        }
        cache_updated = True

        article["ai_image_url"] = url
        article["ai_prompt"]    = prompt
        log.info(f"Image URL generated for: {title[:60]}")
        return article

    # Parallel processing
    results   = [None] * len(articles)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(process, dict(a)): i
            for i, a in enumerate(articles)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                log.error(f"Image enrichment failed for article {idx}: {e}")
                a = dict(articles[idx])
                a["ai_image_url"] = ""
                a["ai_prompt"]    = ""
                results[idx]      = a

    if cache_updated:
        _save_cache(cache)

    return results