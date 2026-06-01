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
    return con.execute(
        "select id, original_path, description, rating, people_count, indexed_at from media where original_path is not null order by indexed_at desc limit ? offset ?",
        (size, offset),
    ).fetchall()


def row_by_id(image_id):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con.execute("select original_path from media where id = ?", (image_id,)).fetchone()


@app.get("/", response_class=HTMLResponse)
def index(page: int = Query(1, ge=1), size: int = Query(3, ge=1)):
    items = ""
    for row in rows(page, size):
        image_id = escape(str(row["id"]))
        path = escape(str(row["original_path"]))
        description = escape(row["description"] or "")
        rating = escape(str(row["rating"] or ""))
        people = "" if row["people_count"] is None else row["people_count"]
        indexed = escape(str(row["indexed_at"] or ""))
        items += f"""
<section class="item">
  <div class="image"><img src="/images/{image_id}" alt="{path}"></div>
  <div class="details">
    <h2>{Path(row['original_path']).name}</h2>
    <p class="path">{path}</p>
    <dl><dt>Rating</dt><dd>{rating}</dd><dt>People</dt><dd>{people}</dd><dt>Indexed</dt><dd>{indexed}</dd></dl>
    <pre>{description}</pre>
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
dl {{ display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px; }}
dt {{ color: #aaa; }}
pre {{ white-space: pre-wrap; line-height: 1.45; font-size: 16px; }}
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
