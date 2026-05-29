from datetime import datetime, timezone

import sqlite_utils


SCHEMA = {
    "source": str,
    "source_id": str,
    "original_path": str,
    "description": str,
    "embedding": bytes,
    "indexed_at": str,
}


def db(path="archive.db"):
    database = sqlite_utils.Database(path)
    database["media"].create(SCHEMA, pk=("source", "source_id"), if_not_exists=True)
    return database


def save(media, description, embedding, path="archive.db"):
    row = {
        "source": media.source,
        "source_id": media.source_id,
        "original_path": str(media.path),
        "description": description,
        "embedding": embedding,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }
    db(path)["media"].upsert(row, pk=("source", "source_id"))
    return row
