import subprocess
from pathlib import Path

import click

import describe
import embed
import sidecar as sidecars
import store
from sources import apple_photos, onedrive


ONEDRIVE_PATH = Path.home() / "Library" / "CloudStorage" / "OneDrive"


def source_media(source):
    if source == "photos":
        return apple_photos.media()
    if source == "onedrive":
        return onedrive.media(ONEDRIVE_PATH)
    return onedrive.media(source)


@click.command()
@click.option("--source", required=True, help="photos, onedrive, or a file/directory path")
@click.option("--db", "db_path", default="archive.db", show_default=True)
@click.option("--backend", default=describe.DEFAULT_BACKEND, show_default=True)
@click.option("--model", default=None)
@click.option("--limit", default=None, type=int)
@click.option("--retries", default=2, show_default=True)
@click.option("--preview", is_flag=True, help="Open each image in Preview.app")
@click.option("--embed/--no-embed", "write_embedding", default=True, show_default=True)
@click.option("--sidecar/--no-sidecar", "write_sidecar", default=True, show_default=True)
def cli(source, db_path, backend, model, limit, retries, preview, write_embedding, write_sidecar):
    for i, media in enumerate(source_media(source), start=1):
        if limit and i > limit:
            return
        click.echo(f"🔎 {media.path}")
        if preview:
            subprocess.run(["open", "-a", "Preview", media.path], check=True)
        text = describe.describe(media.path, backend=backend, model=model, retries=retries)
        vector = embed.embedding_blob(media.path) if write_embedding else None
        store.save(media, text, vector, db_path)
        if write_sidecar:
            click.echo(f"📝 {sidecars.write(media, text)}")
        click.echo("✅ archived")


if __name__ == "__main__":
    cli()
