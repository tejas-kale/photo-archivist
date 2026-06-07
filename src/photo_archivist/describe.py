import base64
import json
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image
from pillow_heif import register_heif_opener


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b")
DEFAULT_BACKEND = os.getenv("VISION_BACKEND", "ollama")
DEFAULT_PROMPT = "Return only JSON with keys: rating keep/review/cull, cull_reason string, focus sharp/acceptable/soft, exposure strong/adequate/poor/clipped, depth_of_field shallow/standard/deep, noise clean/some/heavy, lighting string, time_of_day string, dominant_color_palette string, dominant_colors list, people_count integer, keywords list, description_prose 2-4 detailed sentences describing people, clothing, setting, visible activity, mood, background context, and notable visual details. Do not identify people by name. Avoid private addresses, phone numbers, IDs, or full document text., activity two words."


@dataclass(frozen=True)
class VisionResult:
    rating: str = "review"
    cull_reason: str = ""
    focus: str = "acceptable"
    exposure: str = "adequate"
    depth_of_field: str = "standard"
    noise: str = "clean"
    lighting: str = "unknown"
    time_of_day: str = "unknown"
    dominant_color_palette: str = "unknown"
    dominant_colors: list[str] = field(default_factory=list)
    people_count: int | None = None
    keywords: list[str] = field(default_factory=list)
    description_prose: str = ""
    activity: str = "unknown"

    def get(self, key, default=None):
        aliases = {"description": "description_prose", "number_people": "people_count", "day_night": "time_of_day", "lighting_quality": "lighting"}
        return getattr(self, aliases.get(key, key), default)

    def __getitem__(self, key):
        return self.get(key)


def image_data(path):
    path = Path(path)
    if path.suffix.lower() in {".heic", ".heif"}:
        register_heif_opener()
    image = Image.open(path).convert("RGB")
    image.thumbnail((1280, 1280))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def describe(path, prompt=DEFAULT_PROMPT, backend=DEFAULT_BACKEND, model=None, retries=2):
    prompts = [prompt, f"{prompt}\nReturn valid JSON only.", DEFAULT_PROMPT]
    for attempt in range(retries + 1):
        try:
            text = describe_once(path, prompts[min(attempt, len(prompts) - 1)], backend, model)
            if text:
                return parse(text)
        except (httpx.HTTPError, json.JSONDecodeError):
            pass
    raise RuntimeError("No description after retries")


def parse(text):
    if "{" not in text and "}" not in text:
        return fallback(text)
    if "{" not in text or "}" not in text:
        raise json.JSONDecodeError("truncated JSON", text, 0)
    data = json.loads(text[text.find("{"): text.rfind("}") + 1])
    return VisionResult(
        rating=str(data.get("rating", "review")),
        cull_reason=str(data.get("cull_reason", "")),
        focus=str(data.get("focus", "soft" if data.get("blur") else "acceptable")),
        exposure=str(data.get("exposure", "adequate")),
        depth_of_field=str(data.get("depth_of_field", "standard")),
        noise=str(data.get("noise", "clean")),
        lighting=str(data.get("lighting", data.get("lighting_quality", "unknown"))),
        time_of_day=str(data.get("time_of_day", data.get("day_night", "unknown"))),
        dominant_color_palette=str(data.get("dominant_color_palette", "unknown")),
        dominant_colors=list(data.get("dominant_colors", [])),
        people_count=maybe_int(data.get("people_count", data.get("number_people"))),
        keywords=list(data.get("keywords", [])),
        description_prose=str(data.get("description_prose", data.get("description", ""))),
        activity=str(data.get("activity", "unknown")),
    )


def fallback(text):
    return VisionResult(description_prose=text)


def coerce(data):
    if isinstance(data, VisionResult):
        return data
    return VisionResult(
        lighting=str(data.get("lighting", data.get("lighting_quality", "unknown"))),
        time_of_day=str(data.get("time_of_day", data.get("day_night", "unknown"))),
        people_count=maybe_int(data.get("people_count", data.get("number_people"))),
        description_prose=str(data.get("description_prose", data.get("description", ""))),
        activity=str(data.get("activity", "unknown")),
    )


def maybe_int(value):
    return None if value is None else int(value)


def describe_once(path, prompt=DEFAULT_PROMPT, backend=DEFAULT_BACKEND, model=None):
    if backend == "ollama":
        return describe_ollama(path, prompt, model or OLLAMA_MODEL)
    raise ValueError(f"Unknown backend: {backend}")


def describe_ollama(path, prompt=DEFAULT_PROMPT, model=OLLAMA_MODEL, timeout=600, num_predict=768):
    body = {
        "model": model,
        "prompt": prompt,
        "images": [image_data(path)],
        "stream": False,
        "format": "json",
        "options": {"num_predict": num_predict},
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()
