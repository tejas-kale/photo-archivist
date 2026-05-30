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
    def test_writes_sidecar_for_onedrive(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            media = SourceMedia("onedrive", "rel/IMG_1234.jpeg", image, {})
            meta = PhotoMetadata(100, 80, datetime(2026, 5, 29, tzinfo=timezone.utc), "Apple", "iPhone", 51.5, -0.1, 10.0, 1)
            location = LocationResult("London, UK", "London", "United Kingdom", "gb")
            vision = VisionResult(description_prose="Two lines\nof prose", people_count=2, keywords=["family"], rating="keep", lighting="daylight", time_of_day="day", dominant_colors=["blue"], dominant_color_palette="cool blue")
            face = FaceEmbedding(b"1234", (1, 2, 3, 4), 0.91)
            with unittest.mock.patch.object(sidecar.faces_db, "name_details_for_face", return_value={"name": "Tejas", "source": "predicted", "confidence": 0.88}):
                path = sidecar.write(media, vision, meta, location, [face], [42], datetime(2026, 5, 29, tzinfo=timezone.utc))
            text = path.read_text()

        frontmatter = yaml.safe_load(text.split("---")[1])
        self.assertEqual("IMG_1234.jpeg", frontmatter["file"])
        self.assertEqual("100x80", frontmatter["resolution"])
        self.assertEqual("London, UK", frontmatter["location"]["place"])
        self.assertEqual("high", frontmatter["faces"][0]["detection_quality"])
        self.assertEqual(42, frontmatter["faces"][0]["face_embedding_id"])
        self.assertEqual("Tejas", frontmatter["faces"][0]["person_name"])
        self.assertEqual("predicted", frontmatter["faces"][0]["name_source"])
        self.assertEqual(0.88, frontmatter["faces"][0]["confidence"])
        self.assertEqual("onedrive", frontmatter["source"]["type"])
        self.assertIn("## Description\nTwo lines\nof prose", text)

    def test_sidecar_path_is_beside_image(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            media = SourceMedia("onedrive", "rel/IMG_1234.jpeg", image, {})
            path = sidecar.path_for(media)

        self.assertEqual(image.with_name(f"{image.stem}.description.md"), path)

    def test_refresh_sidecars_updates_face_names(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "IMG_1234.description.md"
            path.write_text("---\nfaces:\n- face_embedding_id: 42\n  bbox: [1, 2, 3, 4]\n---\n\n## Description\ncaption\n")
            with unittest.mock.patch.object(sidecar.faces_db, "name_details_for_face", return_value={"name": "Ishwa", "source": "labelled", "confidence": 1.0}):
                updated = sidecar.refresh_sidecars(Path(d))
            frontmatter = yaml.safe_load(path.read_text().split("---")[1])

        self.assertEqual(1, updated)
        self.assertEqual("Ishwa", frontmatter["faces"][0]["person_name"])
        self.assertEqual("labelled", frontmatter["faces"][0]["name_source"])
        self.assertEqual(1.0, frontmatter["faces"][0]["confidence"])


if __name__ == "__main__":
    unittest.main()
