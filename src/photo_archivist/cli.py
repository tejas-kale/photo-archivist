import subprocess
from pathlib import Path

import click

from photo_archivist import archive_runner, describe, embed, faces, geocode, metadata, ollama_ctl, search, store
from photo_archivist import sidecar as sidecars
from photo_archivist.archive_runner import ArchiveOptions, source_media


def backfill_embeddings(db_path, limit):
    import sqlite3

    con = sqlite3.connect(db_path)
    cols = {row[1] for row in con.execute("pragma table_info(media)")}
    order = "indexed_at desc" if "indexed_at" in cols else "id"
    rows = con.execute(f"select id, original_path from media where embedding is null and original_path is not null order by {order} limit ?", (limit or -1,)).fetchall()
    created = 0
    skipped = 0
    for image_id, path in rows:
        try:
            vector = embed.embedding_blob_subprocess(Path(path))
        except RuntimeError as e:
            click.echo(f"⚠️ embedding skipped {path}: {e}")
            skipped += 1
            continue
        con.execute("update media set embedding = ? where id = ?", (vector, image_id))
        con.commit()
        created += 1
    return created, skipped


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
    options = ArchiveOptions(source, image, db_path, backend, model, limit, retries, preview, write_embedding, embed_subprocess, write_sidecar, write_geocode, write_faces, manage_ollama, restart_ollama_every, cooldown, verbose)
    try:
        for event in archive_runner.archive_events(options, source_func=source_media):
            if event["type"] == "log":
                click.echo(event["message"])
    except ValueError as e:
        raise click.ClickException(str(e)) from e


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


@cli.command("backfill-embeddings")
@click.option("--db", "db_path", default="archive.db", show_default=True)
@click.option("--limit", type=int, default=None)
def backfill_embeddings_cmd(db_path, limit):
    created, skipped = backfill_embeddings(db_path, limit)
    click.echo(f"embeddings: {created} created, {skipped} skipped")


@cli.command("query")
@click.argument("query")
@click.option("--db", "db_path", default="archive.db", show_default=True)
@click.option("--limit", default=50, show_default=True, type=int)
def query_cmd(query, db_path, limit):
    for row in search.find(db_path, query, limit):
        first = row["text"].splitlines()[0] if row["text"] else ""
        click.echo(f"{row['id']}\t{row['original_path']}\t{first}")


@cli.command("serve-ui")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8714, show_default=True)
@click.option("--db", "db_path", default="archive.db", show_default=True)
def serve_ui(host, port, db_path):
    import uvicorn

    from photo_archivist.web import app as webui

    webui.DB_PATH = Path(db_path)
    uvicorn.run(webui.app, host=host, port=port)


@cli.command("train-faces")
@click.option("--min-labels", default=1, show_default=True, type=int)
def train_faces(min_labels):
    faces.train_faces(min_labels=min_labels)
    click.echo("classifier trained")


@cli.command("refresh-sidecars")
@click.argument("path", type=click.Path(path_type=Path), default=Path("."), required=False)
def refresh_sidecars(path):
    updated = sidecars.refresh_sidecars(path)
    click.echo(f"refreshed {updated} sidecars")


if __name__ == "__main__":
    cli()
