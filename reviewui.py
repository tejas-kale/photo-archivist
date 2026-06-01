import sqlite3
from html import escape
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from sources import onedrive


DB_PATH = Path("archive.db")
app = FastAPI()


def rows(page, size):
    size = min(size, 3)
    offset = (page - 1) * size
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cols = {row[1] for row in con.execute("pragma table_info(media)")}
    rating = "rating" if "rating" in cols else "picture_quality" if "picture_quality" in cols else "null"
    people = "people_count" if "people_count" in cols else "number_people" if "number_people" in cols else "null"
    description = "description" if "description" in cols else "''"
    indexed = "indexed_at" if "indexed_at" in cols else "''"
    return con.execute(
        f"select id, original_path, {description} as description, {rating} as rating, {people} as people_count, {indexed} as indexed_at from media where original_path is not null order by indexed_at desc limit ? offset ?",
        (size, offset),
    ).fetchall()


def row_by_id(image_id):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con.execute("select original_path from media where id = ?", (image_id,)).fetchone()


def sidecar_text(path, fallback):
    path = Path(path)
    sidecar = path.with_name(f"{path.stem}.description.md")
    return sidecar.read_text() if sidecar.exists() else fallback or ""


@app.get("/", response_class=HTMLResponse)
def index(page: int = Query(1, ge=1), size: int = Query(3, ge=1)):
    items = ""
    for row in rows(page, size):
        image_id = escape(str(row["id"]))
        path = escape(str(row["original_path"]))
        description = escape(sidecar_text(row["original_path"], row["description"]))
        indexed = escape(str(row["indexed_at"] or ""))
        items += f"""
<section class="item">
  <div class="image"><img src="/images/{image_id}" alt="{path}"></div>
  <div class="details">
    <h2>{Path(row['original_path']).name}</h2>
    <p class="path">{path}</p>
    <p class="indexed">Indexed {indexed}</p>
    <pre class="sidecar">{description}</pre>
  </div>
</section>
"""
    if not items:
        items = '<p class="empty">No archived images.</p>'
    prev_link = f'<a href="/?page={page - 1}&size={min(size, 3)}">Previous</a>' if page > 1 else ""
    next_link = f'<a href="/?page={page + 1}&size={min(size, 3)}">Next</a>'
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Photo Review</title>
<style>
body {{ margin: 0; font-family: system-ui; background: #111; color: #eee; }}
nav {{ position: sticky; top: 0; display: flex; gap: 16px; padding: 12px; background: #181818; border-bottom: 1px solid #333; }}
a {{ color: #8af; }}
.item {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(320px, 1fr); gap: 16px; padding: 16px; border-bottom: 1px solid #333; }}
.image img {{ width: 100%; max-height: 86vh; object-fit: contain; background: #000; }}
.path {{ color: #aaa; overflow-wrap: anywhere; }}
.indexed {{ color: #aaa; }}
.sidecar {{ white-space: pre-wrap; line-height: 1.45; font-size: 14px; max-height: 75vh; overflow: auto; padding: 12px; background: #181818; border: 1px solid #333; border-radius: 6px; }}
.empty {{ padding: 40px; color: #aaa; }}
@media (max-width: 800px) {{ .item {{ grid-template-columns: 1fr; }} }}
</style></head><body><nav>{prev_link}<span>Page {page}</span>{next_link}</nav>{items}</body></html>"""


@app.get("/images/{image_id}")
def image(image_id: str):
    row = row_by_id(image_id)
    if not row:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    path = onedrive.ensure_local(Path(row["original_path"]))
    if not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path)
