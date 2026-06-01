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

    def test_face_storage_is_idempotent_for_source_bbox(self):
        import faces

        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            first = faces.store_face_embeddings("onedrive", "uuid", [face])
            second = faces.store_face_embeddings("onedrive", "uuid", [face])

        self.assertEqual(first, second)

    def test_name_for_face_uses_model_prediction_after_label(self):
        import faces

        labelled = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        unlabelled = faces.FaceEmbedding(np.array([0.99, 0.01], dtype="float32").tobytes(), (5, 6, 7, 8), 0.9)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            faces.store_face_embeddings("onedrive", "a", [labelled])[0]
            unlabelled_id = faces.store_face_embeddings("onedrive", "b", [unlabelled])[0]
            with patch.object(faces, "predict_name", return_value=("Ishwa", 0.88)):
                name = faces.name_for_face(unlabelled_id)

        self.assertEqual("Ishwa", name)

    def test_name_for_face_returns_none_without_model(self):
        import faces

        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            face_id = faces.store_face_embeddings("onedrive", "uuid", [face])[0]
            name = faces.name_for_face(face_id)

        self.assertIsNone(name)

    def test_face_labels_round_trip(self):
        import faces

        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            face_id = faces.store_face_embeddings("onedrive", "uuid", [face])[0]
            faces.label_face(face_id, "Tejas")
            name = faces.name_for_face(face_id)

        self.assertEqual("Tejas", name)

    def test_face_db_stores_and_finds_similar_faces(self):
        import faces

        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (1, 2, 3, 4), 0.95)
        query = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (5, 6, 7, 8), 0.9)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)), patch.object(faces, "detect_faces", return_value=([query], None)):
            ids = faces.store_face_embeddings("onedrive", "uuid", [face])
            rows = faces.find_similar_faces(Path("x.jpg"), top_k=1)

        self.assertEqual([1], ids)
        self.assertEqual("onedrive", rows[0]["source"])
        self.assertAlmostEqual(1.0, rows[0]["cosine_similarity"])
        self.assertEqual([1, 2, 3, 4], rows[0]["bbox"])

    def test_detect_faces_returns_image_array(self):
        import faces
        from PIL import Image as PILImage

        img = PILImage.new("RGB", (100, 80))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.jpg"
            img.save(path)
            with patch.object(faces, "app") as mock_app:
                mock_app.return_value.get.return_value = []
                detections, arr = faces.detect_faces(path)

        self.assertEqual([], detections)
        self.assertEqual((80, 100, 3), arr.shape)

    def test_store_face_embeddings_saves_crops_with_padding(self):
        import faces
        from PIL import Image

        img = np.zeros((80, 100, 3), dtype="uint8")
        img[20:60, 30:70] = 255
        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (30, 20, 70, 60), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            face_id = faces.store_face_embeddings("onedrive", "uuid", [face], img)[0]
            crop_path = faces.crop_path_for(face_id)
            self.assertTrue(crop_path.exists())
            crop = np.array(Image.open(crop_path))
            pad_w = int((70 - 30) * faces.CROP_PADDING)
            pad_h = int((60 - 20) * faces.CROP_PADDING)
            self.assertEqual((40 + 2 * pad_h, 40 + 2 * pad_w), crop.shape[:2])

    def test_crop_clamped_to_image_bounds(self):
        import faces
        from PIL import Image

        img = np.zeros((80, 100, 3), dtype="uint8")
        img[0:30, 80:100] = 255
        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (80, 0, 100, 30), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            face_id = faces.store_face_embeddings("onedrive", "uuid", [face], img)[0]
            crop = np.array(Image.open(faces.crop_path_for(face_id)))
            self.assertEqual(34, crop.shape[0])

    def test_normalized_l2_normalizes_embedding(self):
        import faces

        vec = np.array([3.0, 4.0], dtype="float32")
        result = faces.normalized(vec.tobytes())
        np.testing.assert_allclose([0.6, 0.8], result)

    def test_store_face_embeddings_recreates_missing_crop_for_existing_row(self):
        import faces

        img = np.zeros((80, 100, 3), dtype="uint8")
        face = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (10, 10, 50, 50), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            face_id = faces.store_face_embeddings("onedrive", "uuid", [face], img)[0]
            crop = faces.crop_path_for(face_id)
            crop.unlink()
            second = faces.store_face_embeddings("onedrive", "uuid", [face], img)
            self.assertEqual([face_id], second)
            self.assertTrue(crop.exists())

    def test_backfill_crops_creates_missing_and_skips_unavailable(self):
        import faces
        import logging
        from PIL import Image

        img = np.zeros((80, 100, 3), dtype="uint8")
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            image_path = Path(d) / "available.jpg"
            Image.fromarray(img).save(image_path)
            face_available = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (10, 10, 50, 50), 0.95)
            face_unavailable = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (50, 50, 90, 90), 0.95)
            ids_avail = faces.store_face_embeddings("onedrive", str(image_path), [face_available], img)
            ids_unavail = faces.store_face_embeddings("onedrive", str(Path(d) / "missing.jpg"), [face_unavailable])
            crop_avail = faces.crop_path_for(ids_avail[0])
            crop_avail.unlink()

            with self.assertLogs(level=logging.WARNING) as log:
                created, skipped = faces.backfill_crops()

            self.assertEqual(1, created)
            self.assertEqual(1, skipped)
            self.assertTrue(crop_avail.exists())
            self.assertIn("source unavailable", log.output[0])
    def test_train_faces_defaults_to_high_confidence_threshold(self):
        import pickle
        import faces

        e1 = np.array([1.0, 0.0], dtype="float32")
        e2 = np.array([0.0, 1.0], dtype="float32")
        f1 = faces.FaceEmbedding(e1.tobytes(), (1, 2, 3, 4), 0.95)
        f2 = faces.FaceEmbedding(e2.tobytes(), (5, 6, 7, 8), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            id1 = faces.store_face_embeddings("onedrive", "a", [f1])[0]
            id2 = faces.store_face_embeddings("onedrive", "b", [f2])[0]
            faces.label_face(id1, "Alice")
            faces.label_face(id2, "Bob")
            faces.train_faces()
            with open(faces._classifier_path(), "rb") as f:
                data = pickle.load(f)

        self.assertEqual(0.95, data["threshold"])

    def test_train_predict_round_trip(self):
        import faces

        e1 = np.array([1.0, 0.0], dtype="float32")
        e2 = np.array([0.0, 1.0], dtype="float32")
        e3 = np.array([0.7, 0.7], dtype="float32")
        f1 = faces.FaceEmbedding(e1.tobytes(), (1, 2, 3, 4), 0.95)
        f2 = faces.FaceEmbedding(e2.tobytes(), (5, 6, 7, 8), 0.95)
        f3 = faces.FaceEmbedding(e3.tobytes(), (9, 10, 11, 12), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            id1 = faces.store_face_embeddings("onedrive", "a", [f1])[0]
            id2 = faces.store_face_embeddings("onedrive", "b", [f2])[0]
            id3 = faces.store_face_embeddings("onedrive", "c", [f3])[0]
            faces.label_face(id1, "Alice")
            faces.label_face(id2, "Bob")
            faces.label_face(id3, "Alice")
            faces.train_faces(threshold=0.5)

            name, conf = faces.predict_name(e1.tobytes())
            self.assertEqual("Alice", name)
            self.assertGreater(conf, 0.5)

    def test_predict_name_returns_none_below_threshold(self):
        import faces

        e1 = np.array([1.0, 0.0], dtype="float32")
        e2 = np.array([0.0, 1.0], dtype="float32")
        f1 = faces.FaceEmbedding(e1.tobytes(), (1, 2, 3, 4), 0.95)
        f2 = faces.FaceEmbedding(e2.tobytes(), (5, 6, 7, 8), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            id1 = faces.store_face_embeddings("onedrive", "a", [f1])[0]
            id2 = faces.store_face_embeddings("onedrive", "b", [f2])[0]
            faces.label_face(id1, "Alice")
            faces.label_face(id2, "Bob")
            faces.train_faces()

            unknown = np.array([-1.0, -1.0], dtype="float32")
            name, conf = faces.predict_name(unknown.tobytes())
            self.assertIsNone(name)

    def test_train_faces_raises_when_too_few_labels(self):
        import faces

        e1 = np.array([1.0, 0.0], dtype="float32")
        f1 = faces.FaceEmbedding(e1.tobytes(), (1, 2, 3, 4), 0.95)
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            faces.store_face_embeddings("onedrive", "a", [f1])[0]
            with self.assertRaises(ValueError):
                faces.train_faces()

    def test_train_faces_filters_by_min_labels(self):
        import pickle
        import faces

        e1 = np.array([1.0, 0.0], dtype="float32")
        e2 = np.array([0.0, 1.0], dtype="float32")
        e3 = np.array([0.7, 0.7], dtype="float32")
        with tempfile.TemporaryDirectory() as d, patch.object(faces, "root", return_value=Path(d)):
            ids = [
                faces.store_face_embeddings("onedrive", "a1", [faces.FaceEmbedding(e1.tobytes(), (1, 2, 3, 4), 0.95)])[0],
                faces.store_face_embeddings("onedrive", "a2", [faces.FaceEmbedding(e1.tobytes(), (2, 2, 3, 4), 0.95)])[0],
                faces.store_face_embeddings("onedrive", "b1", [faces.FaceEmbedding(e2.tobytes(), (3, 2, 3, 4), 0.95)])[0],
                faces.store_face_embeddings("onedrive", "b2", [faces.FaceEmbedding(e2.tobytes(), (4, 2, 3, 4), 0.95)])[0],
                faces.store_face_embeddings("onedrive", "c1", [faces.FaceEmbedding(e3.tobytes(), (5, 2, 3, 4), 0.95)])[0],
            ]
            for face_id, name in zip(ids, ["Alice", "Alice", "Bob", "Bob", "Carol"]):
                faces.label_face(face_id, name)
            faces.train_faces(min_labels=2)
            with open(faces._classifier_path(), "rb") as f:
                data = pickle.load(f)

        self.assertEqual(["Alice", "Bob"], data["labels"])
        self.assertEqual(2, data["min_labels"])


if __name__ == "__main__":
    unittest.main()
