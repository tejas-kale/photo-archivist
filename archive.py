import subprocess
from pathlib import Path

import click

import describe
import embed
import faces
import geocode
import metadata
import ollama_ctl
import sidecar as sidecars
import store
from sources import onedrive
from sources.base import SourceMedia


ONEDRIVE_PATH = Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "tejas" / "Pictures"


def source_media(source=None, image=None, limit=None):
    if source == "photos":
        raise ValueError("Apple Photos source is no longer supported. Use OneDrive or a local path.")
    if image:
        path = onedrive.ensure_local(image)
        return [SourceMedia("onedrive", str(path), path, {"path": str(path)})]
    return onedrive.media(ONEDRIVE_PATH if source in (None, "onedrive") else source, limit=limit)


def embedding_blob(path, subprocess_mode):
    return embed.embedding_blob_subprocess(path) if subprocess_mode else embed.embedding_blob(path)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--source", default="onedrive", show_default=True, help="onedrive or a file/directory path")
@click.option("--image", type=click.Path(path_type=Path), help="Archive one image path")
@click.option("--db", "db_path", default="archive.db", show_default=True)
@click.option("--backend", default=describe.DEFAULT_BACKEND, show_default=True)
@click.option("--model", default=None)
@click.option("--limit", default=None, type=int)
@click.option("--retries", default=2, show_default=True)
@click.option("--preview", is_flag=True, help="Open each image in Preview.app")
@click.option("--embed/--no-embed", "write_embedding", default=False, show_default=True)
@click.option("--embed-subprocess/--no-embed-subprocess", default=True, show_default=True)
@click.option("--sidecar/--no-sidecar", "write_sidecar", default=True, show_default=True)
@click.option("--geocode/--no-geocode", "write_geocode", default=True, show_default=True)
@click.option("--faces/--no-faces", "write_faces", default=True, show_default=True)
@click.option("--manage-ollama", is_flag=True)
@click.option("--restart-ollama-every", type=int, default=None)
@click.option("--cooldown", type=float, default=0.0, show_default=True)
@click.option("--verbose", is_flag=True)
def cli(ctx, source, image, db_path, backend, model, limit, retries, preview, write_embedding, embed_subprocess, write_sidecar, write_geocode, write_faces, manage_ollama, restart_ollama_every, cooldown, verbose):
    if ctx.invoked_subcommand:
        return
    if restart_ollama_every and not manage_ollama:
        raise click.ClickException("--restart-ollama-every requires --manage-ollama")
    if manage_ollama:
        ollama_ctl.restart(cooldown)
    processed = 0
    attempted = 0
    for media in source_media(source, image, limit):
        click.echo(f"🔎 {media.path}")
        if preview:
            subprocess.run(["open", "-a", "Preview", media.path], check=True)
        if verbose:
            click.echo("🧾 metadata")
        photo_metadata = metadata.extract_metadata(media.path)
        location = None
        if write_geocode and photo_metadata.gps_lat is not None and photo_metadata.gps_lon is not None:
            if verbose:
                click.echo("🗺️ geocoding")
            location = geocode.reverse_geocode(photo_metadata.gps_lat, photo_metadata.gps_lon)
        if verbose:
            click.echo("🧠 describing")
        try:
            data = describe.coerce(describe.describe(media.path, backend=backend, model=model, retries=retries))
        except RuntimeError as e:
            click.echo(f"⚠️ skipped {media.path}: {e}")
            attempted += 1
            if restart_ollama_every and attempted % restart_ollama_every == 0:
                ollama_ctl.restart(cooldown)
            continue
        if verbose and write_embedding:
            click.echo("🧬 embedding")
        try:
            vector = embedding_blob(media.path, embed_subprocess) if write_embedding else None
        except subprocess.CalledProcessError as e:
            click.echo(f"⚠️ embedding skipped {media.path}: {e}")
            vector = None
        found_faces = []
        face_ids = []
        if write_faces:
            if verbose:
                click.echo("🙂 faces")
            found_faces, image_array = faces.detect_faces(media.path)
            face_ids = faces.store_face_embeddings(media.source, media.source_id, found_faces, image_array)
        if verbose:
            click.echo("💾 saving")
        store.save(media, data, vector, db_path, photo_metadata, location, len(found_faces))
        if write_sidecar:
            click.echo(f"📝 {sidecars.write(media, data, photo_metadata, location, found_faces, face_ids)}")
        click.echo("✅ archived")
        processed += 1
        attempted += 1
        if restart_ollama_every and attempted % restart_ollama_every == 0:
            ollama_ctl.restart(cooldown)
        if limit and processed >= limit:
            break
    if manage_ollama:
        ollama_ctl.stop()


@cli.command("label-face")
@click.argument("face_id", type=int)
@click.argument("name")
def label_face(face_id, name):
    faces.label_face(face_id, name)
    click.echo(f"labelled face {face_id} as {name}")


@cli.command("backfill-crops")
def backfill_crops():
    created, skipped = faces.backfill_crops()
    click.echo(f"backfill: {created} crops created, {skipped} skipped (source unavailable)")


@cli.command("serve-faces")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8714, show_default=True)
def serve_faces(host, port):
    import uvicorn

    from faceui import app

    uvicorn.run(app, host=host, port=port)


@cli.command("train-faces")
def train_faces():
    faces.train_faces()
    click.echo("classifier trained")


@cli.command("refresh-sidecars")
@click.argument("path", type=click.Path(path_type=Path), default=Path("."), required=False)
def refresh_sidecars(path):
    updated = sidecars.refresh_sidecars(path)
    click.echo(f"refreshed {updated} sidecars")


if __name__ == "__main__":
    cli()
