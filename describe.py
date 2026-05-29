import base64
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
MLX_MODEL = os.getenv("MLX_VLM_MODEL", "mlx-community/Qwen3.5-VL-9B-Instruct-4bit")
DEFAULT_BACKEND = os.getenv("VISION_BACKEND", "ollama")
DEFAULT_PROMPT = "Return only JSON with keys: rating keep/review/cull, cull_reason string, focus sharp/acceptable/soft, exposure strong/adequate/poor/clipped, depth_of_field shallow/standard/deep, noise clean/some/heavy, lighting string, time_of_day string, dominant_color_palette string, dominant_colors list, people_count integer, keywords list, description_prose two lines, activity two words."


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
    return base64.b64encode(Path(path).read_bytes()).decode()


def describe(path, prompt=DEFAULT_PROMPT, backend=DEFAULT_BACKEND, model=None, retries=2):
    prompts = [prompt, f"{prompt}\nReturn valid JSON only.", DEFAULT_PROMPT]
    for attempt in range(retries + 1):
        try:
            text = describe_once(path, prompts[min(attempt, len(prompts) - 1)], backend, model)
            if text:
                return parse(text)
        except httpx.HTTPError:
            pass
    raise RuntimeError("No description after retries")


def parse(text):
    if "{" not in text or "}" not in text:
        return fallback(text)
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
    if backend == "mlx-vlm":
        return describe_mlx(path, prompt, model or MLX_MODEL)
    if backend == "ollama":
        return describe_ollama(path, prompt, model or OLLAMA_MODEL)
    raise ValueError(f"Unknown backend: {backend}")


def describe_ollama(path, prompt=DEFAULT_PROMPT, model=OLLAMA_MODEL, timeout=600, num_predict=160):
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


def describe_mlx(path, prompt=DEFAULT_PROMPT, model=MLX_MODEL, max_tokens=160):
    r = subprocess.run(
        ["python", "-m", "mlx_vlm.generate", "--model", model, "--image", str(path), "--prompt", prompt, "--max-tokens", str(max_tokens)],
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()
