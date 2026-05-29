import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PhotoMetadata:
    width: int | None
    height: int | None
    created_at: datetime | None
    camera_make: str | None
    camera_model: str | None
    gps_lat: float | None
    gps_lon: float | None
    gps_altitude_m: float | None
    orientation: int | None


def empty():
    return PhotoMetadata(None, None, None, None, None, None, None, None, None)


def extract_metadata(path: Path) -> PhotoMetadata:
    path = Path(path)
    try:
        r = subprocess.run(["exiftool", "-j", "-n", str(path)], check=True, capture_output=True, text=True)
        data = json.loads(r.stdout or "[]")
        item = data[0] if data else {}
        return PhotoMetadata(
            number(item.get("ImageWidth")),
            number(item.get("ImageHeight")),
            created(item.get("DateTimeOriginal"), path),
            item.get("Make"),
            item.get("Model"),
            decimal(item.get("GPSLatitude")),
            decimal(item.get("GPSLongitude")),
            decimal(item.get("GPSAltitude")),
            number(item.get("Orientation")),
        )
    except FileNotFoundError:
        logging.warning("exiftool not installed")
        return empty()
    except Exception as e:
        logging.warning("could not read metadata for %s: %s", path, e)
        return empty()


def created(value, path):
    if value:
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(value).split("+")[0], fmt)
            except ValueError:
                pass
    try:
        return datetime.fromtimestamp(Path(path).stat().st_mtime)
    except OSError:
        return None


def number(value):
    return None if value is None else int(value)


def decimal(value):
    return None if value is None else float(value)
