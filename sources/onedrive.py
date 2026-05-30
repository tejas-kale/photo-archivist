import random
from pathlib import Path

from sources.base import SourceMedia


EXTENSIONS = {".avif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_local(path):
    path = Path(path).expanduser().resolve()
    path.read_bytes()
    return path


def media(root, limit=None):
    root = Path(root).expanduser().resolve()
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if limit and limit < len(paths):
        paths = random.sample(paths, limit)
    for path in paths:
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            local = ensure_local(path)
            yield SourceMedia("onedrive", str(local), local, {"path": str(local)})
