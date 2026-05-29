from datetime import datetime, timezone
from pathlib import Path

import yaml

import faces as faces_db
from describe import VisionResult
from faces import FaceEmbedding
from geocode import LocationResult
from metadata import PhotoMetadata


def root():
    path = Path.home() / ".photo-archivist"
    path.mkdir(exist_ok=True)
    return path


def path_for(media):
    if media.source == "photos":
        path = root() / "sidecars" / "apple_photos" / f"{media.source_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    return media.path.with_name(f"{media.path.stem}.description.md")


def write(media, vision: VisionResult, photo_metadata: PhotoMetadata, location: LocationResult | None = None, faces: list[FaceEmbedding] | None = None, face_ids: list[int] | None = None, indexed_at: datetime | None = None):
    sidecar_path = path_for(media)
    indexed_at = indexed_at or datetime.now(timezone.utc)
    frontmatter = {
        "file": media.path.name,
        "path": str(media.path.resolve()),
        "parent_folder": media.path.parent.name,
        "resolution": resolution(photo_metadata),
        "size_bytes": media.path.stat().st_size if media.path.exists() else None,
        "creation_time": photo_metadata.created_at.isoformat() if photo_metadata.created_at else None,
        "camera_make": photo_metadata.camera_make,
        "camera_model": photo_metadata.camera_model,
        "orientation": photo_metadata.orientation,
        "rating": vision.rating,
        "cull_reason": vision.cull_reason,
        "technical": {"focus": vision.focus, "exposure": vision.exposure, "depth_of_field": vision.depth_of_field, "noise": vision.noise},
        "lighting": vision.lighting,
        "time_of_day": vision.time_of_day,
        "dominant_color_palette": vision.dominant_color_palette,
        "dominant_colors": vision.dominant_colors,
        "people_count": vision.people_count,
        "keywords": vision.keywords,
        "face_count": len(faces or []),
        "source": {"type": source_type(media.source), "source_id": media.source_id, "album": album(media), "sidecar_path": str(sidecar_path.resolve())},
        "indexed_at": indexed_at.isoformat(),
    }
    if photo_metadata.gps_lat is not None:
        frontmatter["location"] = {"lat": photo_metadata.gps_lat, "lon": photo_metadata.gps_lon, "altitude_m": photo_metadata.gps_altitude_m, "place": location.display_name if location else None}
    if faces:
        frontmatter["faces"] = [face_row(face, face_ids or [], i) for i, face in enumerate(faces)]
    text = "---\n" + yaml.dump(frontmatter, sort_keys=False) + "---\n\n## Description\n" + vision.description_prose + "\n"
    sidecar_path.write_text(text)
    return sidecar_path


def resolution(photo_metadata):
    if photo_metadata.width and photo_metadata.height:
        return f"{photo_metadata.width}x{photo_metadata.height}"
    return None


def source_type(source):
    return "apple_photos" if source == "photos" else "onedrive"


def album(media):
    albums = media.metadata.get("albums") or []
    return albums[0] if albums else None


def face_row(face, ids, index):
    face_id = ids[index] if index < len(ids) else None
    row = {"bbox": list(face.bbox), "detection_quality": quality(face.det_score), "face_embedding_id": face_id, "cluster_id": None}
    name = faces_db.name_for_face(face_id) if face_id else None
    if name:
        row["person_name"] = name
    return row


def quality(score):
    if score >= 0.9:
        return "high"
    if score >= 0.7:
        return "medium"
    return "low"
