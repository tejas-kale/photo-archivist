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
    "camera_make": str,
    "camera_model": str,
    "gps_lat": float,
    "gps_lon": float,
    "place": str,
    "face_count": int,
    "embedding": bytes,
    "indexed_at": str,
}


def db(path="archive.db"):
    database = sqlite_utils.Database(path)
    table = database["media"]
    table.create(SCHEMA, pk="id", if_not_exists=True)
    existing = {row[1] for row in database.conn.execute("pragma table_info(media)")}
    for name, kind in SCHEMA.items():
        if name not in existing:
            database.conn.execute(f"alter table media add column {name} {sql_type(kind)}")
    return database


def sql_type(kind):
    return "blob" if kind is bytes else "real" if kind is float else "integer" if kind in (int, bool) else "text"


def save(media, data, embedding, path="archive.db", photo_metadata=None, location=None, face_count=0):
    row = {
        "id": media.path.stem,
        "source": media.source,
        "source_id": media.source_id,
        "original_path": str(media.path),
        "number_people": data.get("people_count", data.get("number_people")),
        "day_night": data.get("time_of_day", data.get("day_night")),
        "lighting_quality": data.get("lighting", data.get("lighting_quality")),
        "blur": data.get("blur"),
        "picture_quality": data.get("picture_quality"),
        "child": data.get("child"),
        "description": data.get("description_prose", data.get("description")),
        "activity": data.get("activity"),
        "camera_make": getattr(photo_metadata, "camera_make", None),
        "camera_model": getattr(photo_metadata, "camera_model", None),
        "gps_lat": getattr(photo_metadata, "gps_lat", None),
        "gps_lon": getattr(photo_metadata, "gps_lon", None),
        "place": getattr(location, "display_name", None),
        "face_count": face_count,
        "embedding": embedding,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    db(path)["media"].upsert(row, pk="id")
    return row
