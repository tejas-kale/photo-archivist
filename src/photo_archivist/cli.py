import subprocess
from pathlib import Path

import click

from photo_archivist import archive_runner, describe, embed, evaluation, faces, geocode, metadata, ollama_ctl, search, store
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


@cli.group("eval")
def eval_cmd():
    pass


@eval_cmd.command("query-candidates")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=Path("archive.db"), show_default=True)
@click.option("--eval-dir", type=click.Path(path_type=Path), default=evaluation.EVAL_DIR, show_default=True)
def eval_query_candidates(db_path, eval_dir):
    pool = evaluation.query_candidates(db_path, eval_dir)
    click.echo(evaluation.counts_table(pool))


@eval_cmd.command("classify-candidates")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=Path("archive.db"), show_default=True)
@click.option("--eval-dir", type=click.Path(path_type=Path), default=evaluation.EVAL_DIR, show_default=True)
@click.option("--sample-size", default=evaluation.EVAL_SAMPLE, show_default=True, type=int)
@click.option("--target-per-category", default=evaluation.EVAL_TARGET, show_default=True, type=int)
@click.option("--model", default=evaluation.EVAL_CLASSIFIER_MODEL, show_default=True)
def eval_classify_candidates(db_path, eval_dir, sample_size, target_per_category, model):
    pool = evaluation.classify_candidates(db_path, eval_dir, sample_size=sample_size, target_per_category=target_per_category, model=model)
    click.echo(evaluation.counts_table(pool))


@eval_cmd.command("export-candidates")
@click.option("--eval-dir", type=click.Path(path_type=Path), default=evaluation.EVAL_DIR, show_default=True)
@click.option("--output", "output_dir", type=click.Path(path_type=Path), default=None)
@click.option("--limit-per-category", default=30, show_default=True, type=int)
@click.option("--max-size", default=1280, show_default=True, type=int)
@click.option("--quality", default=90, show_default=True, type=int)
def eval_export_candidates(eval_dir, output_dir, limit_per_category, max_size, quality):
    manifest = evaluation.export_candidates(eval_dir, output_dir, limit_per_category, max_size, quality)
    click.echo(f"exported {len(manifest)} candidates")


@eval_cmd.command("import-drafts")
@click.option("--eval-dir", type=click.Path(path_type=Path), default=evaluation.EVAL_DIR, show_default=True)
@click.option("--draft-dir", type=click.Path(path_type=Path), required=True)
def eval_import_drafts(eval_dir, draft_dir):
    count = evaluation.import_drafts(eval_dir, draft_dir)
    click.echo(f"imported {count} drafts")


@eval_cmd.command("score")
@click.option("--eval-dir", type=click.Path(path_type=Path), default=evaluation.EVAL_DIR, show_default=True)
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=None)
@click.option("--backend", default=describe.DEFAULT_BACKEND, show_default=True)
@click.option("--model", default=None)
@click.option("--retries", default=2, show_default=True, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--tracking-uri", default=None)
@click.option("--experiment", default=None)
def eval_score(eval_dir, db_path, backend, model, retries, limit, tracking_uri, experiment):
    try:
        summary = evaluation.score(eval_dir, backend=backend, model=model, retries=retries, limit=limit, db_path=db_path, tracking_uri=tracking_uri, experiment=experiment)
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(evaluation.summary_table(summary))


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
