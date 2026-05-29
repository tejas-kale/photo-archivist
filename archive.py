import subprocess
from dataclasses import replace
from pathlib import Path

import click

import describe
import embed
import faces
import geocode
import metadata
import open_original
import sidecar as sidecars
import store
from sources import apple_photos, onedrive
from sources.base import SourceMedia


ONEDRIVE_PATH = Path.home() / "Library" / "CloudStorage" / "OneDrive"


def source_media(source, image=None):
    if image:
        path = onedrive.ensure_local(image)
        return [SourceMedia("onedrive", str(path), path, {"path": str(path)})]
    if source == "photos":
        return apple_photos.media()
    if source == "onedrive":
        return onedrive.media(ONEDRIVE_PATH)
    return onedrive.media(source)


@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--source", required=False, help="photos, onedrive, or a file/directory path")
@click.option("--image", type=click.Path(path_type=Path), help="Archive one image path")
@click.option("--db", "db_path", default="archive.db", show_default=True)
@click.option("--backend", default=describe.DEFAULT_BACKEND, show_default=True)
@click.option("--model", default=None)
@click.option("--limit", default=None, type=int)
@click.option("--retries", default=2, show_default=True)
@click.option("--preview", is_flag=True, help="Open each image in Preview.app")
@click.option("--embed/--no-embed", "write_embedding", default=True, show_default=True)
@click.option("--sidecar/--no-sidecar", "write_sidecar", default=True, show_default=True)
@click.option("--geocode/--no-geocode", "write_geocode", default=True, show_default=True)
@click.option("--faces/--no-faces", "write_faces", default=True, show_default=True)
@click.option("--verbose", is_flag=True)
def cli(ctx, source, image, db_path, backend, model, limit, retries, preview, write_embedding, write_sidecar, write_geocode, write_faces, verbose):
    if ctx.invoked_subcommand:
        return
    if not source and not image:
        raise click.UsageError("Missing option '--source' or '--image'.")
    for i, media in enumerate(source_media(source, image), start=1):
        if limit and i > limit:
            return
        click.echo(f"🔎 {media.path}")
        if preview:
            subprocess.run(["open", "-a", "Preview", media.path], check=True)
        if verbose:
            click.echo("🧾 metadata")
        photo_metadata = with_source_gps(metadata.extract_metadata(media.path), media)
        location = None
        if write_geocode and photo_metadata.gps_lat is not None and photo_metadata.gps_lon is not None:
            if verbose:
                click.echo("🗺️ geocoding")
            location = geocode.reverse_geocode(photo_metadata.gps_lat, photo_metadata.gps_lon)
        if verbose:
            click.echo("🧠 describing")
        data = describe.coerce(describe.describe(media.path, backend=backend, model=model, retries=retries))
        if verbose and write_embedding:
            click.echo("🧬 embedding")
        vector = embed.embedding_blob(media.path) if write_embedding else None
        found_faces = []
        face_ids = []
        if write_faces:
            if verbose:
                click.echo("🙂 faces")
            found_faces = faces.detect_faces(media.path)
            face_ids = faces.store_face_embeddings(media.source, media.source_id, found_faces)
        if verbose:
            click.echo("💾 saving")
        store.save(media, data, vector, db_path, photo_metadata, location, len(found_faces))
        if write_sidecar:
            click.echo(f"📝 {sidecars.write(media, data, photo_metadata, location, found_faces, face_ids)}")
        click.echo("✅ archived")


@cli.command("label-face")
@click.argument("face_id", type=int)
@click.argument("name")
def label_face(face_id, name):
    faces.label_face(face_id, name)
    click.echo(f"labelled face {face_id} as {name}")


@cli.command("open-photos")
@click.argument("source_id")
def open_photos(source_id):
    open_original.open_original("photos", source_id, None)
    click.echo(f"opened Photos item {source_id}")


def with_source_gps(photo_metadata, media):
    if photo_metadata.gps_lat is not None:
        return photo_metadata
    if media.metadata.get("gps_lat") is None or media.metadata.get("gps_lon") is None:
        return photo_metadata
    return replace(photo_metadata, gps_lat=media.metadata.get("gps_lat"), gps_lon=media.metadata.get("gps_lon"), gps_altitude_m=media.metadata.get("gps_altitude_m"))


if __name__ == "__main__":
    cli()
