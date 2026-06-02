import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class StructuredTests(unittest.TestCase):
    def test_describe_returns_structured_fields(self):
        import describe

        payload = {
            "number_people": 2,
            "day_night": "day",
            "lighting_quality": "soft",
            "blur": False,
            "picture_quality": "good",
            "child": True,
            "description": "A child plays outside.\nAn adult stands nearby.",
            "activity": "playing outside",
        }
        with patch.object(describe, "describe_ollama", return_value=json.dumps(payload)):
            data = describe.describe(Path("x.jpg"), backend="ollama", retries=0)

        self.assertEqual(2, data.people_count)
        self.assertEqual("day", data.time_of_day)
        self.assertEqual("soft", data.lighting)
        self.assertEqual("A child plays outside.\nAn adult stands nearby.", data.description_prose)

    def test_describe_retries_truncated_json_instead_of_preserving_it(self):
        import describe

        payload = json.dumps({"description_prose": "clean prose", "people_count": 1})
        with patch.object(describe, "describe_ollama", side_effect=['{"description_prose": "broken', payload]) as ollama:
            data = describe.describe(Path("x.jpg"), backend="ollama", retries=1)

        self.assertEqual("clean prose", data.description_prose)
        self.assertEqual(2, ollama.call_count)

    def test_describe_preserves_plain_text_when_json_is_missing(self):
        import describe

        text = "A small child is sitting near a window with soft light. The image is slightly blurry."
        with patch.object(describe, "describe_ollama", return_value=text):
            data = describe.describe(Path("x.jpg"), backend="ollama", retries=0)

        self.assertEqual(text, data.description_prose)
        self.assertEqual("unknown", data.time_of_day)
        self.assertIsNone(data.people_count)

    def test_ollama_requests_json_format(self):
        import describe

        image = Path(__file__)
        response = Mock()
        response.json.return_value = {"response": "{}"}
        with patch.object(describe, "image_data", return_value="image"), patch.object(describe.httpx, "post", return_value=response) as post:
            describe.describe_ollama(image)

        body = post.call_args.kwargs["json"]
        self.assertEqual("json", body["format"])
        self.assertEqual("gemma4:e2b", body["model"])
        self.assertEqual(768, body["options"]["num_predict"])

    def test_mlx_backend_is_removed(self):
        import describe

        self.assertFalse(hasattr(describe, "MLX_MODEL"))
        self.assertFalse(hasattr(describe, "describe_mlx"))
        with self.assertRaisesRegex(ValueError, "Unknown backend: mlx-vlm"):
            describe.describe_once(Path("x.jpg"), backend="mlx-vlm")

    def test_image_data_is_converted_resized_jpeg(self):
        import describe

        image = Mock()
        converted = Mock()
        converted.save.side_effect = lambda buf, format, quality: buf.write(b"jpeg")
        image.convert.return_value = converted
        with patch.object(describe, "register_heif_opener") as register, patch.object(describe.Image, "open", return_value=image):
            data = base64.b64decode(describe.image_data(Path("x.heic")))

        register.assert_called_once_with()
        image.convert.assert_called_once_with("RGB")
        converted.thumbnail.assert_called_once_with((1280, 1280))
        converted.save.assert_called_once()
        self.assertEqual(b"jpeg", data)

    def test_jpeg_image_data_is_also_resized(self):
        import describe

        image = Mock()
        converted = Mock()
        converted.save.side_effect = lambda buf, format, quality: buf.write(b"jpeg")
        image.convert.return_value = converted
        with patch.object(describe, "register_heif_opener") as register, patch.object(describe.Image, "open", return_value=image):
            data = base64.b64decode(describe.image_data(Path("x.jpg")))

        register.assert_not_called()
        converted.thumbnail.assert_called_once_with((1280, 1280))
        self.assertEqual(b"jpeg", data)

    def test_store_adds_new_columns_to_existing_database(self):
        import sqlite3
        import store

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "archive.db"
            con = sqlite3.connect(db_path)
            con.execute("create table media (id text primary key, source text)")
            con.commit()
            store.db(db_path)
            cols = {row[1] for row in con.execute("pragma table_info(media)")}

        self.assertIn("camera_make", cols)
        self.assertIn("place", cols)
        self.assertIn("face_count", cols)

    def test_store_uses_filename_stem_as_primary_key(self):
        import sqlite_utils
        import store
        from sources.base import SourceMedia

        data = {
            "number_people": 2,
            "day_night": "day",
            "lighting_quality": "soft",
            "blur": False,
            "picture_quality": "good",
            "child": True,
            "description": "A child plays outside.\nAn adult stands nearby.",
            "activity": "playing outside",
        }
        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "archive.db"
            media = SourceMedia("onedrive", "uuid", Path("/tmp/IMG_1234.jpeg"), {})
            store.save(media, data, None, db_path)
            row = sqlite_utils.Database(db_path)["media"].get("IMG_1234")

        self.assertEqual("IMG_1234", row["id"])
        self.assertEqual(2, row["number_people"])
        self.assertEqual("day", row["day_night"])
        self.assertEqual(1, row["child"])
        self.assertEqual("playing outside", row["activity"])


if __name__ == "__main__":
    unittest.main()
