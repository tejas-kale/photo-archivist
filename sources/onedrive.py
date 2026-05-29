from pathlib import Path

from sources.base import SourceMedia


EXTENSIONS = {".avif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def ensure_local(path):
    path = Path(path).expanduser().resolve()
    path.read_bytes()
    return path


def media(root):
    root = Path(root).expanduser().resolve()
    paths = [ensure_local(root)] if root.is_file() else root.rglob("*")
    for path in paths:
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            local = ensure_local(path)
            yield SourceMedia("onedrive", str(local), local, {"path": str(local)})
