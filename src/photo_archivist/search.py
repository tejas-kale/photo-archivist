import sqlite3
from pathlib import Path


def find(db_path="archive.db", query="", limit=50):
    query = str(query).strip().lower()
    if not query or not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cols = {row[1] for row in con.execute("pragma table_info(media)")}
    fields = [name for name in ["activity", "place"] if name in cols]
    indexed = "indexed_at" if "indexed_at" in cols else "''"
    original = "original_path" if "original_path" in cols else "''"
    description = "description" if "description" in cols else "''"
    extra = ", " + ", ".join(fields) if fields else ""
    rows = con.execute(f"select id, {original} as original_path, {description} as description, {indexed} as indexed_at{extra} from media order by indexed_at desc").fetchall()
    results = []
    for row in rows:
        text = sidecar_text(row["original_path"], row["description"])
        values = [text, row["original_path"]] + [value(row, name) for name in fields]
        haystack = "\n".join(str(v or "") for v in values).lower()
        if query in haystack:
            results.append({"id": str(row["id"]), "original_path": row["original_path"], "description": row["description"], "indexed_at": row["indexed_at"], "text": text})
        if len(results) >= limit:
            break
    return results


def value(row, name):
    try:
        return row[name]
    except (IndexError, KeyError):
        return ""


def sidecar_text(path, fallback=""):
    path = Path(path or "")
    sidecar = path.with_name(f"{path.stem}.description.md")
    return sidecar.read_text() if sidecar.exists() else fallback or ""


def row_by_id(db_path, image_id):
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con.execute("select original_path from media where id = ?", (image_id,)).fetchone()
