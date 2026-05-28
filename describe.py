import base64
import os
from pathlib import Path

import httpx


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_PROMPT = "Describe this image for a searchable personal photo archive. Mention people, objects, setting, visible text, mood, and any useful dates or locations."


def image_data(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def describe_image(path, prompt=DEFAULT_PROMPT, model=None, timeout=120):
    body = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "images": [image_data(path)],
        "stream": False,
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"]


if __name__ == "__main__":
    import sys

    print(describe_image(sys.argv[1]))
