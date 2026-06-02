import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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


@dataclass(frozen=True)
class ArchiveOptions:
    source: str | None = "onedrive"
    image: Path | None = None
    db_path: str = "archive.db"
    backend: str = describe.DEFAULT_BACKEND
    model: str | None = None
    limit: int | None = None
    retries: int = 2
    preview: bool = False
    write_embedding: bool = False
    embed_subprocess: bool = True
    write_sidecar: bool = True
    write_geocode: bool = True
    write_faces: bool = True
    manage_ollama: bool = False
    restart_ollama_every: int | None = None
    cooldown: float = 0.0
    verbose: bool = False
    selection: str = "random"
    start: datetime | None = None
    end: datetime | None = None


def source_media(source=None, image=None, limit=None, selection="random", start=None, end=None):
    if source == "photos":
        raise ValueError("Apple Photos source is no longer supported. Use OneDrive or a local path.")
    if image:
        path = onedrive.ensure_local(image)
        return [SourceMedia("onedrive", str(path), path, {"path": str(path)})]
    root = ONEDRIVE_PATH if source in (None, "onedrive") else source
    return onedrive.media(root, limit=limit, selection=selection, start=start, end=end, hydrate=False)


def embedding_blob(path, subprocess_mode):
    return embed.embedding_blob_subprocess(path) if subprocess_mode else embed.embedding_blob(path)


def archive_events(options: ArchiveOptions, source_func=source_media):
    if options.restart_ollama_every and not options.manage_ollama:
        raise ValueError("--restart-ollama-every requires --manage-ollama")
    if options.manage_ollama:
        yield {"type": "log", "message": "🦙 restarting Ollama"}
        ollama_ctl.restart(options.cooldown)
    processed = 0
    attempted = 0
    items = list(source_func(options.source, options.image, options.limit, options.selection, options.start, options.end))
    if options.limit:
        items = items[:options.limit]
    yield {"type": "total", "total": len(items)}
    for media in items:
        yield {"type": "log", "message": f"🔎 {media.path}"}
        if options.preview:
            subprocess.run(["open", "-a", "Preview", media.path], check=True)
        if options.verbose:
            yield {"type": "log", "message": "🧾 metadata"}
        photo_metadata = metadata.extract_metadata(media.path)
        location = None
        if options.write_geocode and photo_metadata.gps_lat is not None and photo_metadata.gps_lon is not None:
            if options.verbose:
                yield {"type": "log", "message": "🗺️ geocoding"}
            location = geocode.reverse_geocode(photo_metadata.gps_lat, photo_metadata.gps_lon)
        if options.verbose:
            yield {"type": "log", "message": "🧠 describing"}
        try:
            data = describe.coerce(describe.describe(media.path, backend=options.backend, model=options.model, retries=options.retries))
        except RuntimeError as e:
            yield {"type": "log", "message": f"⚠️ skipped {media.path}: {e}"}
            attempted += 1
            if options.restart_ollama_every and attempted % options.restart_ollama_every == 0:
                yield {"type": "log", "message": "🦙 restarting Ollama"}
                ollama_ctl.restart(options.cooldown)
            yield {"type": "progress", "processed": processed, "attempted": attempted}
            continue
        if options.verbose and options.write_embedding:
            yield {"type": "log", "message": "🧬 embedding"}
        try:
            vector = embedding_blob(media.path, options.embed_subprocess) if options.write_embedding else None
        except RuntimeError as e:
            yield {"type": "log", "message": f"⚠️ embedding skipped {media.path}: {e}"}
            vector = None
        found_faces = []
        face_ids = []
        if options.write_faces:
            if options.verbose:
                yield {"type": "log", "message": "🙂 faces"}
            try:
                found_faces, image_array = faces.detect_faces(media.path)
                face_ids = faces.store_face_embeddings(media.source, media.source_id, found_faces, image_array)
            except OSError as e:
                yield {"type": "log", "message": f"⚠️ faces skipped {media.path}: {e}"}
        if options.verbose:
            yield {"type": "log", "message": "💾 saving"}
        store.save(media, data, vector, options.db_path, photo_metadata, location, len(found_faces))
        if options.write_sidecar:
            try:
                yield {"type": "log", "message": f"📝 {sidecars.write(media, data, photo_metadata, location, found_faces, face_ids)}"}
            except OSError as e:
                yield {"type": "log", "message": f"⚠️ sidecar skipped {media.path}: {e}"}
        yield {"type": "log", "message": "✅ archived"}
        processed += 1
        attempted += 1
        if options.restart_ollama_every and attempted % options.restart_ollama_every == 0:
            yield {"type": "log", "message": "🦙 restarting Ollama"}
            ollama_ctl.restart(options.cooldown)
        yield {"type": "progress", "processed": processed, "attempted": attempted}
    if options.manage_ollama:
        yield {"type": "log", "message": "🦙 stopping Ollama"}
        ollama_ctl.stop()
    yield {"type": "done", "processed": processed, "attempted": attempted}
