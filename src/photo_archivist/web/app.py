import random
import threading
from datetime import datetime, time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from photo_archivist import archive_runner, describe, faces, metadata, search, store
from photo_archivist import sidecar as sidecars
from photo_archivist.archive_runner import ArchiveOptions
from photo_archivist.sources import onedrive


DB_PATH = Path("archive.db")
app = FastAPI()
_lock = threading.Lock()
_job = {"status": "idle", "total": 0, "processed": 0, "attempted": 0, "logs": []}


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Photo Archiver</title><style>
body{margin:0;font:18px/1.45 system-ui;background:#101114;color:#eee}button,input,select{font:inherit}h1{font-size:34px;margin:0 0 18px}h2{font-size:24px;margin:0 0 18px}#app{display:grid;grid-template-columns:260px 1fr;min-height:100vh;width:80vw;margin:0 auto}.nav{background:#17191f;border-right:1px solid #30333d;padding:20px;position:sticky;top:0;height:100vh;box-sizing:border-box}.nav button{display:block;width:100%;margin:0 0 10px;padding:13px 14px;border:1px solid #333846;background:#222631;color:#eee;border-radius:10px;text-align:left}.nav button.active{background:#395b9d}.tab{display:none;padding:28px}.tab.active{display:block}.controls{display:grid;grid-template-columns:220px 120px 210px 210px 140px;gap:16px;align-items:end;max-width:980px}.field{display:grid;gap:7px;min-width:0}.span2{grid-column:span 2}input,select{width:100%;min-width:0;background:#181b22;color:#eee;border:1px solid #333846;border-radius:8px;padding:11px 12px;min-height:48px;box-sizing:border-box}.primary{width:100%;background:#3f6db5;color:#fff;border:0;border-radius:8px;padding:12px 18px;min-height:48px}.secondary{background:#222631;color:#eee;border:1px solid #333846;border-radius:8px;padding:12px 18px;min-height:48px}progress{width:100%;height:24px}.logs{background:#050507;border:1px solid #333846;border-radius:10px;padding:14px;max-height:380px;overflow:auto;white-space:pre-wrap;font-size:16px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}.face,.result{background:#181b22;border:1px solid #333846;border-radius:10px;padding:12px}.face img,.result img{width:100%;height:150px;object-fit:cover;background:#000;border-radius:8px}.result{display:grid;grid-template-columns:210px 1fr;gap:14px}.result img{height:180px}.muted{color:#aaa}.toolbar{display:flex;justify-content:space-between;align-items:center;margin:0 0 18px}.toolbar button{width:160px}.hidden{display:none!important}@media(max-width:760px){#app{grid-template-columns:1fr;width:100vw}.nav{height:auto;position:static}.result{grid-template-columns:1fr}.span2{grid-column:auto}}
</style></head><body><div id="app"><div class="nav"><h2>Photo Archiver</h2><button class="active" data-tab="archive">Archive</button><button data-tab="faces">Faces</button><button data-tab="search">Search</button></div><main>
<section id="archive" class="tab active"><h1>Archive photos</h1><div class="controls"><label class="field">Source<select id="sourceChoice"><option value="onedrive">OneDrive</option><option value="local">Local folder</option></select></label><label id="sourcePathWrap" class="field span2 hidden">Local path<input id="sourcePath" placeholder="/Users/you/Pictures/export"></label><label class="field">Images<input id="limit" type="number" min="1" value="10"></label><label class="field">Model<select id="model"><option value="">Default</option><option value="gemma4:e2b">gemma4:e2b</option><option value="llava:latest">llava:latest</option></select></label><label class="field">Photos<select id="selection"><option value="random">Random</option><option value="latest">Latest first</option><option value="period">Within period</option></select></label><label id="startWrap" class="field hidden">From<input id="start" type="date"></label><label id="endWrap" class="field hidden">To<input id="end" type="date"></label><button id="startJob" class="primary">Start</button></div><p><progress id="bar" value="0" max="1"></progress> <span id="status">idle</span></p><details open><summary>Logs</summary><pre id="logs" class="logs"></pre></details></section>
<section id="faces" class="tab"><h1>Faces</h1><div class="toolbar"><button id="loadFaces" class="primary">Refresh</button><button id="saveFaces" class="primary">Save</button></div><div id="facesGrid" class="grid"></div></section>
<section id="search" class="tab"><h1>Search</h1><div class="row"><input id="q" placeholder="search descriptions"><button id="runSearch" class="primary">Search</button></div><div id="results"></div></section>
</main></div><script>
const $=id=>document.getElementById(id);let timer=null;
document.querySelectorAll('.nav button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.nav button,.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');$(b.dataset.tab).classList.add('active')});
function source(){return $('sourceChoice').value=='local'?$('sourcePath').value:'onedrive'}
function payload(){let period=$('selection').value=='period';return{source:source(),limit:Number($('limit').value)||null,model:$('model').value||null,selection:period?'latest':$('selection').value,start:period?$('start').value||null:null,end:period?$('end').value||null:null}}
function toggleFields(){$('sourcePathWrap').classList.toggle('hidden',$('sourceChoice').value!='local');let period=$('selection').value=='period';$('startWrap').classList.toggle('hidden',!period);$('endWrap').classList.toggle('hidden',!period)}
$('sourceChoice').onchange=toggleFields;$('selection').onchange=toggleFields;toggleFields();
$('startJob').onclick=async()=>{if(!source()){$('logs').textContent='Choose a local path.';return}$('startJob').disabled=true;$('status').textContent='starting';let r=await fetch('/api/archive/start',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(payload())});if(!r.ok){$('logs').textContent=(await r.text());$('startJob').disabled=false;return}poll();timer=setInterval(poll,700)};
async function poll(){let r=await fetch('/api/archive/status'),j=await r.json();$('status').textContent=j.status+' '+j.processed+'/'+j.total;$('bar').max=j.total||1;$('bar').value=j.processed;$('logs').textContent=j.logs.join('\\n');if(j.status!='running'&&timer){clearInterval(timer);timer=null;$('startJob').disabled=false}}
$('loadFaces').onclick=loadFaces;
async function loadFaces(){let j=await (await fetch('/api/faces')).json();$('facesGrid').innerHTML=j.faces.map(f=>`<div class="face"><img src="/api/faces/${f.id}.jpg"><input data-face="${f.id}" list="names" placeholder="name" value="${f.name||''}"></div>`).join('')+'<datalist id="names">'+j.names.map(n=>`<option value="${n}">`).join('')+'</datalist>'}
$('saveFaces').onclick=async()=>{let labels={};document.querySelectorAll('[data-face]').forEach(i=>{if(i.value.trim())labels[i.dataset.face]=i.value.trim()});await fetch('/api/faces/labels',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(labels)});loadFaces()};
$('runSearch').onclick=runSearch;$('q').onkeydown=e=>{if(e.key=='Enter')runSearch()};
async function runSearch(){let j=await (await fetch('/api/search?q='+encodeURIComponent($('q').value))).json();$('results').innerHTML=j.results.map(r=>`<section class="result"><img src="/api/images/${r.id}"><div><h3>${esc(r.original_path.split('/').pop())}</h3><p class="muted">${esc(r.original_path)}</p><pre>${esc(r.text)}</pre></div></section>`).join('')||'<p class="muted">No matches.</p>'}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
poll();
</script></body></html>"""


def reset_job():
    global _job
    with _lock:
        _job = {"status": "idle", "total": 0, "processed": 0, "attempted": 0, "logs": []}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.post("/api/archive/start")
async def start_archive(request: Request):
    data = await request.json()
    with _lock:
        if _job["status"] == "running":
            return JSONResponse({"detail": "archive job already running"}, status_code=409)
        _job.update({"status": "running", "total": 0, "processed": 0, "attempted": 0, "logs": ["Starting archive job…"]})
    options = ArchiveOptions(source=data.get("source") or "onedrive", db_path=str(DB_PATH), model=data.get("model") or None, limit=data.get("limit"), write_embedding=True, manage_ollama=True, restart_ollama_every=25, cooldown=5, verbose=True, selection=data.get("selection") or "random", start=parse_day(data.get("start"), False), end=parse_day(data.get("end"), True))
    threading.Thread(target=run_job, args=(options,), daemon=True).start()
    return {"status": "running"}


@app.get("/api/archive/status")
def archive_status():
    with _lock:
        return dict(_job)


def run_job(options):
    try:
        for event in archive_runner.archive_events(options, source_func=archive_runner.source_media):
            with _lock:
                if event["type"] == "total":
                    _job["total"] = event["total"]
                if event["type"] == "log":
                    _job["logs"].append(event["message"])
                if event["type"] == "progress":
                    _job["processed"] = event["processed"]
                    _job["attempted"] = event["attempted"]
                if event["type"] == "done":
                    _job["status"] = "done"
    except Exception as e:
        with _lock:
            _job["status"] = "failed"
            _job["logs"].append(str(e))


def parse_day(value, end):
    if not value:
        return None
    day = datetime.fromisoformat(value).date()
    return datetime.combine(day, time.max if end else time.min)


@app.get("/api/search")
def search_api(q: str = "", limit: int = Query(50, ge=1, le=200)):
    return {"results": search.find(DB_PATH, q, limit)}


@app.get("/api/images/{image_id}")
def image(image_id: str):
    row = search.row_by_id(DB_PATH, image_id)
    if not row:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    path = onedrive.ensure_local(Path(row["original_path"]))
    if not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path)


@app.get("/api/faces")
def face_items(size: int = Query(20, ge=1, le=100)):
    con = faces.db()
    labelled = {row[0] for row in con.execute("select face_id from face_labels")}
    all_faces = con.execute("select id from faces order by id").fetchall()
    unlabelled = [row[0] for row in all_faces if row[0] not in labelled and faces.crop_path_for(row[0]).exists()]
    sample = random.sample(unlabelled, min(size, len(unlabelled))) if unlabelled else []
    names = sorted({row[0] for row in con.execute("select distinct name from face_labels")})
    items = []
    for face_id in sample:
        details = faces.name_details_for_face(face_id)
        items.append({"id": face_id, "name": details["name"] if details and details.get("source") == "predicted" else "", "confidence": details.get("confidence") if details else None})
    return {"faces": items, "names": names}


@app.post("/api/faces/labels")
async def label_faces(request: Request):
    labels = await request.json()
    count = 0
    for key, value in labels.items():
        name = str(value).strip()
        if name:
            faces.label_face(int(key), name)
            count += 1
    return {"labelled": count}


@app.get("/api/faces/{face_id}.jpg")
def face_crop(face_id: int):
    path = faces.crop_path_for(face_id)
    if not path.exists():
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")
