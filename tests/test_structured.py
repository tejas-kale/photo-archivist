import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
        response = unittest.mock.Mock()
        response.json.return_value = {"response": "{}"}
        with patch.object(describe.httpx, "post", return_value=response) as post:
            describe.describe_ollama(image)

        self.assertEqual("json", post.call_args.kwargs["json"]["format"])

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
            media = SourceMedia("photos", "uuid", Path("/tmp/IMG_1234.jpeg"), {})
            store.save(media, data, None, db_path)
            row = sqlite_utils.Database(db_path)["media"].get("IMG_1234")

        self.assertEqual("IMG_1234", row["id"])
        self.assertEqual(2, row["number_people"])
        self.assertEqual("day", row["day_night"])
        self.assertEqual(1, row["child"])
        self.assertEqual("playing outside", row["activity"])


if __name__ == "__main__":
    unittest.main()
