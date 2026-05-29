import json
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np


class MetadataGeocodeFacesTests(unittest.TestCase):
    def test_extract_metadata_parses_exiftool_json(self):
        import metadata

        payload = [{
            "ImageWidth": 4032,
            "ImageHeight": 3024,
            "DateTimeOriginal": "2026:05:29 12:34:56",
            "Make": "Apple",
            "Model": "iPhone 15 Pro",
            "GPSLatitude": 51.5,
            "GPSLongitude": -0.12,
            "GPSAltitude": 42.0,
            "Orientation": 1,
        }]
        run = Mock(stdout=json.dumps(payload))
        with patch.object(metadata.subprocess, "run", return_value=run):
            data = metadata.extract_metadata(Path("x.jpg"))

        self.assertEqual(4032, data.width)
        self.assertEqual(3024, data.height)
        self.assertEqual(datetime(2026, 5, 29, 12, 34, 56), data.created_at)
        self.assertEqual("Apple", data.camera_make)
        self.assertEqual(51.5, data.gps_lat)

    def test_extract_metadata_missing_exiftool_returns_empty(self):
        import metadata

        with patch.object(metadata.subprocess, "run", side_effect=FileNotFoundError):
            data = metadata.extract_metadata(Path("x.jpg"))

        self.assertIsNone(data.width)
        self.assertIsNone(data.created_at)

    def test_reverse_geocode_uses_cache(self):
        import geocode

        with tempfile.TemporaryDirectory() as d, patch.object(geocode, "root", return_value=Path(d)):
            first = Mock(raw={"address": {"city": "London", "country": "United Kingdom", "country_code": "gb"}}, address="London, UK")
            locator = Mock()
            locator.reverse.return_value = first
            with patch.object(geocode, "Nominatim", return_value=locator), patch.object(geocode.time, "sleep") as sleep:
                result = geocode.reverse_geocode(51.501, -0.123)
                cached = geocode.reverse_geocode(51.50104, -0.12304)

        self.assertEqual("London, UK", result.display_name)
        self.assertEqual("London", cached.city)
        locator.reverse.assert_called_once_with((51.501, -0.123), exactly_one=True, timeout=10)
        sleep.assert_called_once_with(1.1)

    def test_face_db_stores_and_finds_similar_faces(self):
        import faces

        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        query = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (5, 6, 7, 8), 0.9)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)), patch.object(faces, "detect_faces", return_value=[query]):
            ids = faces.store_face_embeddings("photos", "uuid", [face])
            rows = faces.find_similar_faces(Path("x.jpg"), top_k=1)

        self.assertEqual([1], ids)
        self.assertEqual("photos", rows[0]["source"])
        self.assertAlmostEqual(1.0, rows[0]["cosine_similarity"])
        self.assertEqual([1, 2, 3, 4], rows[0]["bbox"])


if __name__ == "__main__":
    unittest.main()
