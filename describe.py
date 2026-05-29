import base64
import os
import subprocess
from pathlib import Path

import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
MLX_MODEL = os.getenv("MLX_VLM_MODEL", "mlx-community/Qwen3.5-VL-9B-Instruct-4bit")
DEFAULT_BACKEND = os.getenv("VISION_BACKEND", "ollama")
DEFAULT_PROMPT = "Describe this image for a searchable personal photo archive. Mention people, objects, setting, visible text, mood, and any useful dates or locations."


def image_data(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def describe(path, prompt=DEFAULT_PROMPT, backend=DEFAULT_BACKEND, model=None, retries=2):
    prompts = [prompt, f"{prompt}\nAnswer in one concise sentence.", "Describe the image in one concise sentence."]
    for attempt in range(retries + 1):
        try:
            text = describe_once(path, prompts[min(attempt, len(prompts) - 1)], backend, model)
            if text:
                return text
        except httpx.HTTPError:
            pass
    raise RuntimeError("No description after retries")


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
