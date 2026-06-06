import hashlib
import json
import os
import random
import shutil
import sqlite3
import tempfile
from dataclasses import asdict, fields
from pathlib import Path

import httpx
import mlflow
import yaml
from PIL import Image
from pillow_heif import register_heif_opener

from photo_archivist import archive_runner, describe
from photo_archivist.archive_runner import ArchiveOptions
from photo_archivist.sources import onedrive
from photo_archivist.sources.base import SourceMedia
from photo_archivist.sources.onedrive import EXTENSIONS


CATEGORIES = ["well-lit-outdoor", "indoor-low-light", "people", "visible-text", "bad-exposure-blurry", "unusual-subjects"]
EVAL_DIR = Path(os.getenv("PHOTO_ARCHIVIST_EVAL_DIR", "eval"))
EVAL_TARGET = int(os.getenv("PHOTO_ARCHIVIST_EVAL_TARGET", "30"))
EVAL_SAMPLE = int(os.getenv("PHOTO_ARCHIVIST_EVAL_SAMPLE", "200"))
EVAL_CLASSIFIER_MODEL = os.getenv("PHOTO_ARCHIVIST_EVAL_CLASSIFIER_MODEL", "gemma4:e2b")
EVAL_READ_SIDECARS = os.getenv("PHOTO_ARCHIVIST_EVAL_READ_SIDECARS", "0") == "1"
TERMS = {
    "well-lit-outdoor": ["outdoor", "outside", "sun", "sunny", "daylight", "bright", "sky", "park", "garden", "beach", "mountain", "landscape", "trees", "street"],
    "indoor-low-light": ["indoor", "inside", "room", "home", "kitchen", "restaurant", "dim", "dark", "low light", "night", "evening", "lamp", "shadow"],
    "people": ["person", "people", "portrait", "group", "family", "child", "adult", "face", "selfie", "crowd", "friends"],
    "visible-text": ["text", "sign", "writing", "words", "poster", "menu", "label", "document", "receipt", "book", "screen", "logo", "notice"],
    "bad-exposure-blurry": ["blurry", "blur", "soft", "out of focus", "overexposed", "underexposed", "clipped", "dark", "poor", "bad exposure", "cull", "noisy"],
    "unusual-subjects": ["food", "meal", "dinner", "lunch", "document", "receipt", "paper", "pet", "dog", "cat", "abstract", "pattern", "art", "screenshot"],
}


def image_paths(eval_dir):
    root = Path(eval_dir) / "images"
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def golden_path(eval_dir, image):
    return Path(eval_dir) / "golden" / f"{Path(image).stem}.json"


def load_golden(eval_dir, image):
    beside = Path(image).with_suffix(".json")
    path = beside if beside.exists() else golden_path(eval_dir, image)
    return json.loads(path.read_text()) if path.exists() else None


def source_for(paths):
    def source_func(*args):
        return [SourceMedia("eval", path.stem, path, {"path": str(path)}) for path in paths]
    return source_func


def pool_path(eval_dir=EVAL_DIR):
    return Path(eval_dir) / "candidates.json"


def empty_pool():
    return {"categories": {category: [] for category in CATEGORIES}, "skipped": []}


def load_pool(eval_dir=EVAL_DIR):
    path = pool_path(eval_dir)
    if not path.exists():
        return empty_pool()
    pool = json.loads(path.read_text())
    pool.setdefault("skipped", [])
    pool.setdefault("categories", {})
    for category in CATEGORIES:
        pool["categories"].setdefault(category, [])
    return pool


def save_pool(eval_dir, pool):
    path = pool_path(eval_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, indent=2))
    return pool


def export_candidates(eval_dir=EVAL_DIR, output_dir=None, limit_per_category=30, max_size=1280, quality=90):
    register_heif_opener()
    output_dir = Path(output_dir or Path(eval_dir) / "upload")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    pool = load_pool(eval_dir)
    skipped = set(pool.get("skipped", []))
    for category in CATEGORIES:
        outdir = output_dir / category
        outdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for item in pool["categories"].get(category, []):
            if item["id"] in skipped or n >= limit_per_category:
                continue
            source = Path(item["path"]).expanduser()
            if not source.exists():
                continue
            rel = Path(category) / f"{item['id']}.jpg"
            image = Image.open(source).convert("RGB")
            image.thumbnail((max_size, max_size))
            image.save(output_dir / rel, "JPEG", quality=quality, optimize=True)
            manifest.append({"id": item["id"], "category": category, "file": rel.as_posix(), "original_path": str(source)})
            n += 1
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def import_drafts(eval_dir=EVAL_DIR, draft_dir=None):
    draft_dir = Path(draft_dir or Path(eval_dir) / "drafts")
    dest = Path(eval_dir) / "drafts"
    count = 0
    for path in draft_dir.rglob("*.json"):
        rel = path.relative_to(draft_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text())
        count += 1
    return count


def draft_path(eval_dir, category, item_id):
    return Path(eval_dir) / "drafts" / category / f"{item_id}.json"


def draft_for_candidate(eval_dir, category, item_id):
    path = draft_path(eval_dir, category, item_id)
    return json.loads(path.read_text()) if path.exists() else None


def candidate_id(path):
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def archive_rows(db_path):
    if not Path(db_path).exists():
        return []
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cols = {row[1] for row in con.execute("pragma table_info(media)")}
    if "original_path" not in cols:
        return []
    return con.execute("select * from media where original_path is not null").fetchall()


def query_candidates(db_path="archive.db", eval_dir=EVAL_DIR):
    pool = load_pool(eval_dir)
    found = {category: [] for category in CATEGORIES}
    for row in archive_rows(db_path):
        front, text = sidecar_data(value(row, "original_path")) if EVAL_READ_SIDECARS else ({}, "")
        haystack = row_text(row, front, text)
        for category in CATEGORIES:
            score = category_score(category, row, front, haystack)
            if score:
                found[category].append(candidate(row, category, "db", score))
    for category, items in found.items():
        merge(pool, category, sorted(items, key=lambda c: c["score"], reverse=True))
    return save_pool(eval_dir, pool)


def classify_candidates(db_path="archive.db", eval_dir=EVAL_DIR, sample_size=EVAL_SAMPLE, target_per_category=EVAL_TARGET, model=EVAL_CLASSIFIER_MODEL):
    pool = load_pool(eval_dir)
    needed = {category for category in CATEGORIES if len(pool["categories"].get(category, [])) < target_per_category}
    if not needed:
        return pool
    seen = {item["id"] for items in pool["categories"].values() for item in items}
    rows = [row for row in archive_rows(db_path) if candidate_id(value(row, "original_path")) not in seen]
    random.shuffle(rows)
    for row in rows[:sample_size]:
        if not needed:
            break
        path = Path(value(row, "original_path"))
        try:
            category = classify_image(path, model=model)
        except (OSError, RuntimeError, httpx.HTTPError):
            continue
        if category in needed:
            merge(pool, category, [candidate(row, category, "ollama", 1)])
            if len(pool["categories"][category]) >= target_per_category:
                needed.remove(category)
    return save_pool(eval_dir, pool)


def classify_image(path, model=EVAL_CLASSIFIER_MODEL, timeout=180, num_predict=16):
    prompt = "Return exactly one category label and nothing else. Labels: " + ", ".join(CATEGORIES) + ". Food, documents, pets, and abstract images are unusual-subjects."
    body = {"model": model, "prompt": prompt, "images": [describe.image_data(path)], "stream": False, "options": {"num_predict": num_predict}}
    r = httpx.post(f"{describe.OLLAMA_URL}/api/generate", json=body, timeout=timeout)
    r.raise_for_status()
    return parse_category(r.json()["response"])


def parse_category(text):
    text = str(text).strip().strip('"\'`').lower().replace("_", "-")
    if text.startswith("{") and text.endswith("}"):
        data = json.loads(text)
        text = str(data.get("category", data.get("label", ""))).lower().replace("_", "-")
    text = text.replace(" ", "-")
    for category in CATEGORIES:
        if category in text:
            return category
    if "outdoor" in text:
        return "well-lit-outdoor"
    if "indoor" in text or "low-light" in text:
        return "indoor-low-light"
    if "people" in text or "person" in text:
        return "people"
    if "text" in text:
        return "visible-text"
    if "blurry" in text or "exposure" in text:
        return "bad-exposure-blurry"
    if "unusual" in text:
        return "unusual-subjects"
    raise RuntimeError(f"Unknown category: {text}")


def sidecar_data(path):
    path = Path(path or "")
    sidecar = path.with_name(f"{path.stem}.description.md")
    if not sidecar.exists():
        return {}, ""
    text = sidecar.read_text()
    parts = text.split("---", 2)
    front = yaml.safe_load(parts[1]) if len(parts) > 2 else {}
    return front or {}, text


def row_text(row, front, sidecar_text):
    values = [value(row, name) for name in ["description", "description_prose", "activity", "place", "day_night", "lighting_quality", "picture_quality", "rating", "keywords"]]
    values += [front.get("rating"), front.get("lighting"), front.get("time_of_day"), front.get("keywords"), front.get("technical")]
    values.append(sidecar_text)
    return "\n".join(str(v or "") for v in values).lower()


def category_score(category, row, front, text):
    score = sum(1 for term in TERMS[category] if term in text)
    if category == "well-lit-outdoor" and value(row, "day_night") == "day":
        score += 2
    if category == "indoor-low-light" and (value(row, "day_night") == "night" or "poor" in str(value(row, "lighting_quality") or "")):
        score += 2
    if category == "people" and (num(value(row, "number_people")) > 0 or num(value(row, "face_count")) > 0 or num(front.get("people_count")) > 0):
        score += 4
    if category == "bad-exposure-blurry" and (num(value(row, "blur")) or value(row, "picture_quality") == "poor" or front.get("rating") == "cull"):
        score += 4
    return score


def candidate(row, category, source, score):
    path = value(row, "original_path")
    return {"id": candidate_id(path), "archive_id": str(value(row, "id")), "path": str(path), "category": category, "source": source, "score": score}


def merge(pool, category, items):
    existing = {item["id"] for item in pool["categories"].setdefault(category, [])}
    for item in items:
        if item["id"] not in existing:
            pool["categories"][category].append(item)
            existing.add(item["id"])


def value(row, name):
    try:
        return row[name]
    except (IndexError, KeyError):
        return None


def num(value):
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    return int(float(text)) if text.replace(".", "", 1).isdigit() else 0


def candidate_counts(pool):
    return {category: len(pool.get("categories", {}).get(category, [])) for category in CATEGORIES}


def counts_table(pool):
    counts = candidate_counts(pool)
    return "\n".join(["category             candidates"] + [f"{category.ljust(20)} {counts[category]}" for category in CATEGORIES])


def labelled_count(eval_dir, category):
    root = Path(eval_dir) / "images" / category
    return len(list(root.glob("*.json"))) if root.exists() else 0


def labelled_originals(eval_dir, category):
    root = Path(eval_dir) / "images" / category
    paths = set()
    for file in root.glob("*.json") if root.exists() else []:
        data = json.loads(file.read_text())
        if data.get("original_path"):
            paths.add(data["original_path"])
    return paths


def find_candidate(eval_dir, category, item_id):
    for item in load_pool(eval_dir)["categories"].get(category, []):
        if item["id"] == item_id:
            return item
    return None


def next_candidate(eval_dir=EVAL_DIR, category="people", target=EVAL_TARGET):
    pool = load_pool(eval_dir)
    skipped = set(pool.get("skipped", []))
    labelled = labelled_originals(eval_dir, category)
    count = labelled_count(eval_dir, category)
    item = None
    if count < target:
        for candidate_row in pool["categories"].get(category, []):
            if candidate_row["id"] not in skipped and candidate_row["path"] not in labelled:
                item = dict(candidate_row)
                draft = draft_for_candidate(eval_dir, category, item["id"])
                if draft:
                    item["draft"] = draft
                break
    return {"categories": CATEGORIES, "category": category, "candidate": item, "labelled": count, "target": target}


def skip_candidate(eval_dir, category, item_id):
    pool = load_pool(eval_dir)
    if item_id not in pool["skipped"]:
        pool["skipped"].append(item_id)
    save_pool(eval_dir, pool)
    return pool["skipped"]


def label_candidate(eval_dir, category, item_id, rating, keywords, description_prose):
    item = find_candidate(eval_dir, category, item_id)
    if not item:
        raise ValueError("Candidate not found")
    source = onedrive.ensure_local(Path(item["path"]))
    dest = Path(eval_dir) / "images" / category / source.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    draft = draft_for_candidate(eval_dir, category, item_id) or {}
    data = dict(draft)
    if draft.get("source"):
        data["draft_source"] = draft["source"]
    data.update({"category": category, "rating": rating, "keywords": keyword_list(keywords), "description_prose": description_prose, "original_path": item["path"], "candidate_id": item_id, "source": "human-reviewed"})
    dest.with_suffix(".json").write_text(json.dumps(data, indent=2))
    return data


def keyword_list(value):
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def score(eval_dir="eval", backend=describe.DEFAULT_BACKEND, model=None, retries=2, limit=None, db_path=None, tracking_uri=None, experiment=None):
    eval_dir = Path(eval_dir)
    all_images = image_paths(eval_dir)
    loaded = [(p, load_golden(eval_dir, p)) for p in all_images]
    missing_goldens = sum(1 for _, golden in loaded if not golden)
    pairs = [(p, g) for p, g in loaded if g]
    if limit:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError("No golden records found")
    paths = [p for p, _ in pairs]
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment = experiment or os.getenv("MLFLOW_EXPERIMENT", "photo-archivist-eval")
    db_path = db_path or eval_dir / "archive.db"
    options = ArchiveOptions(source=str(eval_dir / "images"), db_path=str(db_path), backend=backend, model=model, retries=retries, write_embedding=False, write_geocode=False, write_faces=False, write_sidecar=False, limit=None, selection="latest")
    outputs = {}
    for event in archive_runner.archive_events(options, source_func=source_for(paths)):
        if event["type"] == "vision":
            outputs[Path(event["path"]).stem] = asdict(event["data"])
    rows = [row(path, golden, outputs.get(path.stem)) for path, golden in pairs]
    summary = summarise(rows, missing_goldens)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with tempfile.TemporaryDirectory() as d, mlflow.start_run(run_name=f"eval-{backend}-{model or describe.OLLAMA_MODEL}"):
        root = Path(d)
        (root / "results.json").write_text(json.dumps({"summary": summary, "images": rows}, indent=2, default=str))
        (root / "summary.md").write_text(summary_table(summary) + "\n")
        mlflow.log_params({"backend": backend, "model": model or describe.OLLAMA_MODEL, "eval_dir": str(eval_dir), "db_path": str(db_path), "retries": retries})
        mlflow.log_metrics(summary)
        mlflow.log_artifacts(root, artifact_path="eval")
    return summary


def row(path, golden, output):
    if not output:
        return {"image": str(path), "golden": golden, "output": None, "rating_accuracy": 0.0, "keyword_coverage": 0.0, "field_completeness": 0.0, "failed": True}
    return {
        "image": str(path),
        "golden": golden,
        "output": output,
        "rating_accuracy": rating_accuracy(golden, output),
        "keyword_coverage": keyword_coverage(golden, output),
        "field_completeness": field_completeness(output),
        "failed": False,
    }


def rating_accuracy(golden, output):
    return 1.0 if norm(output.get("rating")) == norm(golden.get("rating")) else 0.0


def keyword_coverage(golden, output):
    golden_words = {norm(k) for k in golden.get("keywords", []) if norm(k)}
    output_words = {norm(k) for k in output.get("keywords", []) if norm(k)}
    return 1.0 if not golden_words else len(golden_words & output_words) / len(golden_words)


def field_completeness(output):
    names = [f.name for f in fields(describe.VisionResult)]
    return sum(1 for name in names if output.get(name) is not None) / len(names)


def norm(value):
    return str(value or "").strip().lower()


def summarise(rows, missing_goldens):
    n = len(rows) or 1
    return {
        "images_scored": len(rows),
        "images_missing_goldens": missing_goldens,
        "images_failed": sum(1 for r in rows if r["failed"]),
        "rating_accuracy": sum(r["rating_accuracy"] for r in rows) / n,
        "keyword_coverage": sum(r["keyword_coverage"] for r in rows) / n,
        "field_completeness": sum(r["field_completeness"] for r in rows) / n,
    }


def summary_table(summary):
    rows = [(key, value) for key, value in summary.items()]
    width = max(len(key) for key, _ in rows)
    return "\n".join(["metric".ljust(width) + "  value"] + [key.ljust(width) + f"  {display(value)}" for key, value in rows])


def display(value):
    return f"{value:.3f}" if isinstance(value, float) else str(value)
