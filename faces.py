import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FaceEmbedding:
    embedding: bytes
    bbox: tuple[int, int, int, int]
    det_score: float


def root():
    path = Path.home() / ".photo-archivist"
    path.mkdir(exist_ok=True)
    return path


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


def detect_faces(path: Path) -> list[FaceEmbedding]:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
        image = np.array(Image.open(path).convert("RGB"))
        return [face_embedding(face) for face in app().get(image)]
    except ImportError:
        logging.warning("insightface not installed")
        return []


def face_embedding(face):
    bbox = tuple(int(x) for x in face.bbox)
    vector = np.array(face.embedding, dtype="float32")
    return FaceEmbedding(vector.tobytes(), bbox, float(face.det_score))


def store_face_embeddings(source: str, source_id: str, faces: list[FaceEmbedding]) -> list[int]:
    con = db()
    ids = []
    for face in faces:
        row = con.execute("select id from faces where source = ? and source_id = ? and bbox_x1 = ? and bbox_y1 = ? and bbox_x2 = ? and bbox_y2 = ?", (source, source_id, *face.bbox)).fetchone()
        if row:
            ids.append(row[0])
            continue
        values = (source, source_id, face.embedding, *face.bbox, face.det_score, datetime.now(timezone.utc).isoformat())
        cur = con.execute("insert into faces (source, source_id, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, indexed_at) values (?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        ids.append(cur.lastrowid)
    con.commit()
    return ids


def label_face(face_id: int, name: str):
    db().execute("insert or replace into face_labels values (?, ?, ?)", (face_id, name, datetime.now(timezone.utc).isoformat())).connection.commit()


def name_for_face(face_id: int, threshold=0.7) -> str | None:
    con = db()
    row = con.execute("select name from face_labels where face_id = ?", (face_id,)).fetchone()
    if row:
        return row[0]
    return inferred_name(con, face_id, threshold)


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
    query = detect_faces(query_path)
    if not query:
        return []
    con = db()
    rows = con.execute("select source, source_id, embedding, bbox_x1, bbox_y1, bbox_x2, bbox_y2 from faces").fetchall()
    if not rows:
        return []
    q = np.frombuffer(query[0].embedding, dtype="float32")
    matrix = np.vstack([np.frombuffer(row[2], dtype="float32") for row in rows])
    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    order = np.argsort(-sims)[:top_k]
    return [result(rows[i], float(sims[i])) for i in order]


def result(row, score):
    return {"source": row[0], "source_id": row[1], "cosine_similarity": score, "bbox": [row[3], row[4], row[5], row[6]]}
