from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import faces as face_db

app = FastAPI()

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE = (TEMPLATE_DIR / "grid.html").read_text()


@app.get("/", response_class=HTMLResponse)
def grid(page: int = Query(1, ge=1), size: int = Query(20, ge=1, le=100)):
    con = face_db.db()
    labelled = {row[0] for row in con.execute("select face_id from face_labels")}
    all_faces = con.execute("select id from faces order by id").fetchall()
    unlabelled = [row[0] for row in all_faces if row[0] not in labelled]
    start = (page - 1) * size
    page_faces = unlabelled[start:start + size]
    names = sorted({row[0] for row in con.execute("select distinct name from face_labels")})
    items = ""
    for fid in page_faces:
        items += f'<div class="face-cell"><img src="/faces/{fid}.jpg" loading="lazy"><input name="face_{fid}" list="names" placeholder="name"></div>\n'
    if not items:
        items = '<p class="empty">No unlabelled faces.</p>'
    return TEMPLATE.replace("{{items}}", items).replace("{{names}}", "\n".join(f'<option value="{n}">' for n in names))


@app.post("/label")
async def label(request: Request):
    form = await request.form()
    for key in form:
        name = str(form[key]).strip()
        if not key.startswith("face_") or not name:
            continue
        face_id = int(key.split("_", 1)[1])
        face_db.label_face(face_id, name)
    return {"ok": True}


@app.get("/faces/{filename}.jpg")
def serve_crop(filename: str):
    path = face_db.crop_path_for(int(filename))
    if not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/names")
def names():
    con = face_db.db()
    names = sorted({row[0] for row in con.execute("select distinct name from face_labels")})
    return {"names": names}
