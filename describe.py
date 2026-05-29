import base64
import os
import subprocess
from pathlib import Path

import click
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


@click.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--prompt", default=DEFAULT_PROMPT, show_default=False)
@click.option("--model", default=None, help=f"Default: {DEFAULT_MODEL}")
@click.option("--timeout", default=120, show_default=True)
@click.option("--preview", is_flag=True, help="Open the image in Preview.app")
def cli(image, prompt, model, timeout, preview):
    click.echo(f"🔎 Reading {image}")
    if preview:
        click.echo("🖼️ Opening Preview")
        subprocess.run(["open", "-a", "Preview", image], check=True)
    click.echo(f"🧠 Asking {model or DEFAULT_MODEL}")
    click.echo("📝 Description:")
    click.echo(describe_image(image, prompt, model, timeout))
    click.echo("✅ Done")


if __name__ == "__main__":
    cli()
