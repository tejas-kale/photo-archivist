import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image


CROP_PADDING = 0.15


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: bytes
    bbox: tuple[int, int, int, int]
    det_score: float


def root():
    path = Path.home() / ".photo-archivist"
    path.mkdir(exist_ok=True)
    return path


def faces_dir():
    path = root() / "faces"
    path.mkdir(exist_ok=True)
    return path


def crop_path_for(face_id: int) -> Path:
    return faces_dir() / f"{face_id}.jpg"


def db():
    con = sqlite3.connect(root() / "faces.db")
    con.execute("create table if not exists faces (id integer primary key, source text not null, source_id text not null, embedding blob not null, bbox_x1 int, bbox_y1 int, bbox_x2 int, bbox_y2 int, det_score real, indexed_at text)")
    con.execute("create unique index if not exists faces_source_bbox on faces (source, source_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2)")
    con.execute("create table if not exists face_labels (face_id integer primary key, name text not null, labelled_at text)")
    return con


@cache
def app():
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=0)
    return face_app


def detect_faces(path: Path) -> tuple[list[FaceEmbedding], np.ndarray | None]:
    from pillow_heif import register_heif_opener

    register_heif_opener()
    image = np.array(Image.open(path).convert("RGB"))
    detections = [face_embedding(face) for face in app().get(image)]
    return detections, image


def face_embedding(face):
    bbox = tuple(int(x) for x in face.bbox)
    vector = np.array(face.embedding, dtype="float32")
    return FaceEmbedding(vector.tobytes(), bbox, float(face.det_score))


def save_crop(image: np.ndarray, bbox: tuple[int, int, int, int], face_id: int):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    pad_w = int(bw * CROP_PADDING)
    pad_h = int(bh * CROP_PADDING)
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)
    crop = image[cy1:cy2, cx1:cx2]
    Image.fromarray(crop).save(crop_path_for(face_id))


def store_face_embeddings(source: str, source_id: str, faces: list[FaceEmbedding], image_array: np.ndarray | None = None) -> list[int]:
    con = db()
    ids = []
    for face in faces:
        row = con.execute("select id from faces where source = ? and source_id = ? and bbox_x1 = ? and bbox_y1 = ? and bbox_x2 = ? and bbox_y2 = ?", (source, source_id, *face.bbox)).fetchone()
        if row:
            face_id = row[0]
            ids.append(face_id)
            if image_array is not None and not crop_path_for(face_id).exists():
                save_crop(image_array, face.bbox, face_id)
            continue
        values = (source, source_id, face.embedding, *face.bbox, face.det_score, datetime.now(timezone.utc).isoformat())
        cur = con.execute("insert into faces (source, source_id, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, indexed_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        face_id = cur.lastrowid
        ids.append(face_id)
        if image_array is not None:
            save_crop(image_array, face.bbox, face_id)
    con.commit()
    return ids


def label_face(face_id: int, name: str):
    db().execute("insert or replace into face_labels values (?, ?, ?)", (face_id, name, datetime.now(timezone.utc).isoformat())).connection.commit()


def normalized(embedding_bytes: bytes) -> np.ndarray:
    vec = np.frombuffer(embedding_bytes, dtype="float32")
    return vec / np.linalg.norm(vec)


def name_for_face(face_id: int) -> str | None:
    details = name_details_for_face(face_id)
    return details["name"] if details else None


def name_details_for_face(face_id: int) -> dict | None:
    con = db()
    row = con.execute("select name from face_labels where face_id = ?", (face_id,)).fetchone()
    if row:
        return {"name": row[0], "source": "labelled", "confidence": 1.0}
    face = con.execute("select embedding from faces where id = ?", (face_id,)).fetchone()
    if not face:
        return None
    name, confidence = predict_name(face[0])
    if not name:
        return None
    return {"name": name, "source": "predicted", "confidence": confidence}


def inferred_name(con, face_id, threshold):
    face = con.execute("select embedding from faces where id = ?", (face_id,)).fetchone()
    rows = con.execute("select faces.embedding, face_labels.name from faces join face_labels on faces.id = face_labels.face_id").fetchall()
    if not face or not rows:
        return None
    q = np.frombuffer(face[0], dtype="float32")
    matrix = np.vstack([np.frombuffer(row[0], dtype="float32") for row in rows])
    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    index = int(np.argmax(sims))
    return rows[index][1] if sims[index] >= threshold else None


def find_similar_faces(query_path: Path, top_k: int = 10) -> list[dict]:
    detections, _ = detect_faces(query_path)
    if not detections:
        return []
    con = db()
    rows = con.execute("select source, source_id, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2 from faces").fetchall()
    if not rows:
        return []
    q = np.frombuffer(detections[0].embedding, dtype="float32")
    matrix = np.vstack([np.frombuffer(row[2], dtype="float32") for row in rows])
    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    order = np.argsort(-sims)[:top_k]
    return [result(rows[i], float(sims[i])) for i in order]


def result(row, score):
    return {"source": row[0], "source_id": row[1], "cosine_similarity": score, "bbox": [row[3], row[4], row[5], row[6]]}


def _classifier_path():
    return root() / "face_classifier.pkl"


def train_faces(threshold: float = 0.95, min_labels: int = 1):
    import pickle

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    con = db()
    rows = con.execute(
        "select faces.embedding, face_labels.name from faces join face_labels on faces.id = face_labels.face_id"
    ).fetchall()
    counts = {name: sum(1 for row in rows if row[1] == name) for name in {row[1] for row in rows}}
    rows = [row for row in rows if counts[row[1]] >= min_labels]
    if len(rows) < 2:
        raise ValueError("Need at least 2 labelled faces to train a classifier")
    names = [r[1] for r in rows]
    if len(set(names)) < 2:
        raise ValueError("Need at least 2 distinct labels to train a classifier")
    X = np.vstack([normalized(r[0]) for r in rows])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_scaled, names)
    with open(_classifier_path(), "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "labels": model.classes_.tolist(), "threshold": threshold, "normalised": True, "min_labels": min_labels}, f)


def predict_name(embedding_bytes: bytes) -> tuple[str | None, float]:
    import pickle

    path = _classifier_path()
    if not path.exists():
        return None, 0.0
    with open(path, "rb") as f:
        data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    threshold = data["threshold"]
    vec = normalized(embedding_bytes).reshape(1, -1)
    vec_scaled = scaler.transform(vec)
    probs = model.predict_proba(vec_scaled)[0]
    idx = int(np.argmax(probs))
    if probs[idx] < threshold:
        return None, float(probs[idx])
    return str(model.classes_[idx]), float(probs[idx])


def backfill_crops() -> tuple[int, int]:
    con = db()
    rows = con.execute("select id, source_id from faces").fetchall()
    created = 0
    skipped = 0
    for face_id, source_id in rows:
        crop = crop_path_for(face_id)
        if crop.exists():
            continue
        path = Path(source_id)
        if not path.exists():
            logging.warning("crop backfill skipped: source unavailable for face %d (%s)", face_id, source_id)
            skipped += 1
            continue
        from pillow_heif import register_heif_opener
        register_heif_opener()
        image = np.array(Image.open(path).convert("RGB"))
        face_row = con.execute("select bbox_x1, bbox_y1, bbox_x2, bbox_y2 from faces where id = ?", (face_id,)).fetchone()
        bbox = (face_row[0], face_row[1], face_row[2], face_row[3])
        save_crop(image, bbox, face_id)
        created += 1
    return created, skipped
