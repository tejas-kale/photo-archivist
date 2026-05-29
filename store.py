from datetime import datetime, timezone

import sqlite_utils


SCHEMA = {
    "id": str,
    "source": str,
    "source_id": str,
    "original_path": str,
    "number_people": int,
    "day_night": str,
    "lighting_quality": str,
    "blur": bool,
    "picture_quality": str,
    "child": bool,
    "description": str,
    "activity": str,
    "embedding": bytes,
    "indexed_at": str,
}


def db(path="archive.db"):
    database = sqlite_utils.Database(path)
    database["media"].create(SCHEMA, pk="id", if_not_exists=True)
    return database


def save(media, data, embedding, path="archive.db"):
    row = {
        "id": media.path.stem,
        "source": media.source,
        "source_id": media.source_id,
        "original_path": str(media.path),
        "number_people": data.get("number_people"),
        "day_night": data.get("day_night"),
        "lighting_quality": data.get("lighting_quality"),
        "blur": data.get("blur"),
        "picture_quality": data.get("picture_quality"),
        "child": data.get("child"),
        "description": data.get("description"),
        "activity": data.get("activity"),
        "embedding": embedding,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    db(path)["media"].upsert(row, pk="id")
    return row
