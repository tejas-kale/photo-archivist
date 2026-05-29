from pathlib import Path

import osxphotos

from sources.base import SourceMedia


def photo_path(photo):
    paths = [photo.path, *(photo.path_derivatives or [])]
    return next((Path(p) for p in paths if p and Path(p).is_file()), None)


def metadata(photo):
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
    }


def media():
    for photo in osxphotos.PhotosDB().photos():
        path = photo_path(photo)
        if path:
            yield SourceMedia("photos", photo.uuid, path, metadata(photo))
