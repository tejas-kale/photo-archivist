import base64
import mimetypes
import os
from pathlib import Path

import httpx


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
DEFAULT_PROMPT = "Describe this image for a searchable personal photo archive. Mention people, objects, setting, visible text, mood, and any useful dates or locations."


def image_url(path):
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def headers(api_key=None):
    key = api_key or os.environ["OPENROUTER_API_KEY"]
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "photo-archivist"),
    }


def completion(messages, model=None, api_key=None, timeout=120):
    body = {"model": model or DEFAULT_MODEL, "messages": messages}
    r = httpx.post(OPENROUTER_URL, headers=headers(api_key), json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def describe_image(path, prompt=DEFAULT_PROMPT, model=None, api_key=None):
    return completion([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url(path)}},
            ],
        }
    ], model, api_key)


if __name__ == "__main__":
    import sys

    print(describe_image(sys.argv[1]))
