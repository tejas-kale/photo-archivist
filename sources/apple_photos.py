import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import osxphotos
from osxphotos import PhotoInfo, PhotosDB
from osxphotos.exportoptions import ExportOptions
from osxphotos.photoexporter import PhotoExporter
import sqlite_utils

from sources.base import SourceMedia

log = logging.getLogger(__name__)
temporary_exports = set()


def candidate_path(photo: PhotoInfo) -> Path | None:
    if photo.path:
        return Path(photo.path)
    if hasattr(photo, "_path_5"):
        path = photo._path_5()
        return Path(path) if path else None
    if hasattr(photo, "_path_4"):
        path = photo._path_4()
        return Path(path) if path else None
    info = getattr(photo, "_info", {})
    db = getattr(photo, "_db", None)
    directory = info.get("directory")
    filename = info.get("filename")
    masters = getattr(db, "_masters_path", None)
    if directory and filename and str(directory).startswith("/"):
        return Path(directory) / filename
    if directory and filename and masters:
        return Path(masters) / directory / filename
    return None


def is_stub(photo: PhotoInfo) -> bool:
    path = candidate_path(photo)
    if path is None:
        return True
    if not path.exists():
        return True
    size = path.stat().st_size
    if size == 0:
        return True
    if not hasattr(os, "getxattr"):
        return False
    try:
        os.getxattr(path, "com.apple.icloud.itemName")
    except OSError:
        return False
    return size == 0


def stderr(error):
    value = error.stderr
    if isinstance(value, bytes):
        return value.decode(errors="replace").strip()
    return str(value or "").strip()


def is_photos_library_path(path: Path) -> bool:
    return any(part.endswith(".photoslibrary") for part in path.parts)


def brctl(action: str, path: Path) -> bool:
    try:
        subprocess.run(["brctl", action, str(path)], check=True, capture_output=True)
        return True
    except FileNotFoundError as error:
        raise RuntimeError("brctl was not found on PATH; iCloud file pinning requires macOS brctl") from error
    except subprocess.CalledProcessError as error:
        if action == "evict":
            log.warning("Could not evict %s with brctl: %s", path, stderr(error))
        else:
            log.warning("brctl %s failed for %s: %s", action, path, stderr(error))
        return False


def export_local(photo: PhotoInfo, timeout_s: int) -> Path:
    start = time.monotonic()
    directory = Path(tempfile.mkdtemp(prefix="photo-archivist-icloud-"))
    options = ExportOptions(download_missing=True, use_photokit=True, overwrite=True, increment=False, timeout=timeout_s)
    results = PhotoExporter(photo).export(directory, filename=photo.filename, options=options)
    if results.error:
        shutil.rmtree(directory, ignore_errors=True)
        raise TimeoutError(f"Photos could not export {photo.uuid} from iCloud: {results.error[0][1]}")
    if not results.exported:
        shutil.rmtree(directory, ignore_errors=True)
        missing = f": missing {results.missing[0]}" if results.missing else ""
        raise TimeoutError(f"Photos did not export {photo.uuid} from iCloud{missing}")
    path = Path(results.exported[0])
    temporary_exports.add(path)
    elapsed = time.monotonic() - start
    log.info("Downloaded %s from iCloud via PhotoKit (%.1fs)", photo.filename, elapsed)
    return path


def release_local(path: Path) -> None:
    path = Path(path)
    if path in temporary_exports:
        temporary_exports.discard(path)
        if path.exists():
            path.unlink()
        parent = path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
        return
    evict_local(path)


def ensure_local(photo: PhotoInfo, timeout_s: int = 120, poll_interval_s: float = 2.0) -> Path:
    path = candidate_path(photo)
    if path and not is_stub(photo):
        return path
    if path is None:
        raise TimeoutError(f"iCloud photo has no library path: {photo.uuid}")

    if is_photos_library_path(path):
        return export_local(photo, timeout_s)

    start = time.monotonic()
    if not brctl("download", path):
        return export_local(photo, timeout_s)
    warned = False

    while True:
        if path.exists() and path.stat().st_size > 0 and not is_stub(photo):
            elapsed = time.monotonic() - start
            log.info("Downloaded %s from iCloud via brctl (%.1fs)", photo.filename, elapsed)
            return path
        elapsed = time.monotonic() - start
        if elapsed >= timeout_s:
            raise TimeoutError(f"Timed out downloading {path} from iCloud")
        if elapsed > 30 and not warned:
            log.warning("Still downloading %s from iCloud via brctl (%.1fs)", path.name, elapsed)
            warned = True
        time.sleep(poll_interval_s)


def evict_local(path: Path) -> None:
    path = Path(path)
    if not path.exists() or is_photos_library_path(path):
        return
    brctl("evict", path)


def indexed_source_ids(db: sqlite_utils.Database):
    columns = {row[1] for row in db.conn.execute("pragma table_info(media)")}
    if "source_type" in columns:
        yield from db.query("select source_id from media where source_type = 'apple_photos'")
    elif "source" in columns:
        yield from db.query("select source_id from media where source in ('apple_photos', 'photos')")


def evict_already_indexed(db: sqlite_utils.Database, photos_db: PhotosDB) -> None:
    for row in indexed_source_ids(db):
        photo = photos_db.get_photo(row["source_id"])
        if photo and photo.path and not is_stub(photo):
            evict_local(Path(photo.path))


def photo_path(photo: PhotoInfo):
    paths = [photo.path, *(photo.path_derivatives or [])]
    return next((Path(p) for p in paths if p and Path(p).is_file()), None)


def metadata(photo: PhotoInfo):
    return {
        "uuid": photo.uuid,
        "filename": photo.filename,
        "date": photo.date.isoformat() if photo.date else None,
        "title": photo.title,
        "description": photo.description,
        "keywords": photo.keywords,
        "albums": photo.albums,
        "persons": photo.persons,
        "path": str(photo.path) if photo.path else None,
        "gps_lat": getattr(photo, "latitude", None),
        "gps_lon": getattr(photo, "longitude", None),
        "gps_altitude_m": getattr(photo, "altitude", None),
    }


def iter_photos(photos, evict_after: bool = True, limit: int | None = None, max_consecutive_download_failures: int = 10):
    attempts = 0
    failures = 0
    for photo in photos:
        if limit is not None and attempts >= limit:
            break
        attempts += 1
        path = None
        try:
            path = ensure_local(photo)
            failures = 0
            yield SourceMedia("photos", photo.uuid, path, metadata(photo))
        except TimeoutError as error:
            failures += 1
            log.warning("Could not download %s from iCloud: %s", getattr(photo, "filename", photo.path), error)
            if max_consecutive_download_failures and failures >= max_consecutive_download_failures:
                log.warning("Stopped Apple Photos import after %s consecutive iCloud download failures", failures)
                break
        finally:
            if evict_after and path:
                release_local(path)


def media(evict_after: bool = True, db: sqlite_utils.Database | None = None, photos_db: PhotosDB | None = None, limit: int | None = None):
    photos_db = photos_db or osxphotos.PhotosDB()
    if db:
        evict_already_indexed(db, photos_db)
    yield from iter_photos(photos_db.photos(), evict_after=evict_after, limit=limit)
