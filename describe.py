import base64
import os
import random
import subprocess
from pathlib import Path

import click
import httpx
import osxphotos


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
DEFAULT_PROMPT = "Describe this image for a searchable personal photo archive. Mention people, objects, setting, visible text, mood, and any useful dates or locations."


def image_data(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def describe_image(path, prompt=DEFAULT_PROMPT, model=None, timeout=600, num_predict=160):
    body = {
        "model": model or DEFAULT_MODEL,
        "prompt": prompt,
        "images": [image_data(path)],
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()["response"].strip()


def describe_with_retries(path, prompt, model, timeout, num_predict, retries):
    prompts = [
        prompt,
        f"{prompt}\nAnswer in one concise sentence.",
        "Describe the image in one concise sentence.",
    ]
    for attempt in range(1, retries + 2):
        click.echo(f"🔁 Attempt {attempt}/{retries + 1}")
        try:
            text = describe_image(path, prompts[min(attempt - 1, len(prompts) - 1)], model, timeout, num_predict)
            if text:
                return text
            click.echo("⚠️ Empty response")
        except httpx.HTTPError as e:
            click.echo(f"⚠️ {type(e).__name__}: {e}")
    raise click.ClickException("No description after retries")


def photo_path(photo):
    paths = [photo.path, *(photo.path_derivatives or [])]
    return next((Path(p) for p in paths if p and Path(p).is_file()), None)


def random_photo():
    photos = osxphotos.PhotosDB().photos()
    choices = [(p, photo_path(p)) for p in photos if not p.ismovie]
    choices = [(p, path) for p, path in choices if path]
    if not choices:
        raise click.ClickException("No local Photos images found")
    return random.choice(choices)


@click.command()
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=False)
@click.option("--model", default=None, help=f"Default: {DEFAULT_MODEL}")
@click.option("--timeout", default=600, show_default=True)
@click.option("--num-predict", default=160, show_default=True)
@click.option("--retries", default=2, show_default=True)
@click.option("--preview", is_flag=True, help="Open the image in Preview.app")
def cli(prompt, model, timeout, num_predict, retries, preview):
    photo, image = random_photo()
    click.echo(f"🎲 Picked {photo.uuid}")
    click.echo(f"🔎 Reading {image}")
    if preview:
        click.echo("🖼️ Opening Preview")
        subprocess.run(["open", "-a", "Preview", image], check=True)
    click.echo(f"🧠 Asking {model or DEFAULT_MODEL}")
    click.echo(f"⏱️ Timeout {timeout}s, max tokens {num_predict}")
    click.echo(f"🛟 Retries {retries}")
    click.echo("📝 Description:")
    click.echo(describe_with_retries(image, prompt, model, timeout, num_predict, retries))
    click.echo("✅ Done")


if __name__ == "__main__":
    cli()
