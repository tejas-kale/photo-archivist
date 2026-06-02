import random
from datetime import datetime
from pathlib import Path

from sources.base import SourceMedia


EXTENSIONS = {".avif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_local(path):
    path = Path(path).expanduser().resolve()
    path.read_bytes()
    return path


def media(root, limit=None, selection="random", start: datetime | None = None, end: datetime | None = None, hydrate=True):
    root = Path(root).expanduser().resolve()
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    paths = [p for p in paths if in_period(p, start, end)]
    if selection == "latest":
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
    if selection == "random" and limit and limit < len(paths):
        paths = random.sample(paths, limit)
    elif limit:
        paths = paths[:limit]
    for path in paths:
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            local = ensure_local(path) if hydrate else path
            yield SourceMedia("onedrive", str(local), local, {"path": str(local)})


def in_period(path, start, end):
    mtime = path.stat().st_mtime
    if start and mtime < start.timestamp():
        return False
    if end and mtime > end.timestamp():
        return False
    return True
