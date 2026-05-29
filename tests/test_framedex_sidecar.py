import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import yaml

from describe import VisionResult
from faces import FaceEmbedding
from geocode import LocationResult
from metadata import PhotoMetadata
from sources.base import SourceMedia


class FramedexSidecarTests(unittest.TestCase):
    def test_writes_framedex_sidecar_for_onedrive(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            media = SourceMedia("onedrive", "rel/IMG_1234.jpeg", image, {})
            meta = PhotoMetadata(100, 80, datetime(2026, 5, 29, tzinfo=timezone.utc), "Apple", "iPhone", 51.5, -0.1, 10.0, 1)
            location = LocationResult("London, UK", "London", "United Kingdom", "gb")
            vision = VisionResult(description_prose="Two lines\nof prose", people_count=2, keywords=["family"], rating="keep", lighting="daylight", time_of_day="day", dominant_colors=["blue"], dominant_color_palette="cool blue")
            face = FaceEmbedding(b"1234", (1, 2, 3, 4), 0.91)
            with unittest.mock.patch.object(sidecar.faces_db, "name_for_face", return_value="Tejas"):
                path = sidecar.write(media, vision, meta, location, [face], [42], datetime(2026, 5, 29, tzinfo=timezone.utc))
            text = path.read_text()

        frontmatter = yaml.safe_load(text.split("---")[1])
        self.assertEqual("IMG_1234.jpeg", frontmatter["file"])
        self.assertEqual("100x80", frontmatter["resolution"])
        self.assertEqual("London, UK", frontmatter["location"]["place"])
        self.assertEqual("high", frontmatter["faces"][0]["detection_quality"])
        self.assertEqual(42, frontmatter["faces"][0]["face_embedding_id"])
        self.assertEqual("Tejas", frontmatter["faces"][0]["person_name"])
        self.assertEqual("onedrive", frontmatter["source"]["type"])
        self.assertIn("## Description\nTwo lines\nof prose", text)

    def test_writes_apple_photos_sidecar_under_home(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(sidecar, "root", return_value=Path(d)):
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            media = SourceMedia("photos", "uuid", image, {"albums": ["Trip"]})
            meta = PhotoMetadata(None, None, None, None, None, None, None, None, None)
            vision = VisionResult(description_prose="caption")
            path = sidecar.write(media, vision, meta)
            self.assertTrue(path.exists())

        self.assertEqual(Path(d) / "sidecars" / "apple_photos" / "uuid.md", path)


if __name__ == "__main__":
    unittest.main()
