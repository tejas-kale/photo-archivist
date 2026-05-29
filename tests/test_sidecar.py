import tempfile
import unittest
from pathlib import Path

from sources.base import SourceMedia


class SidecarTests(unittest.TestCase):
    def test_sidecar_omits_null_and_empty_metadata(self):
        import sidecar

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            media = SourceMedia("photos", "uuid", image, {"description": None, "keywords": [], "persons": [], "title": None, "date": "2026-05-29"})
            path = sidecar.write(media, "caption")
            text = path.read_text()

        self.assertNotIn("description: null", text)
        self.assertNotIn("keywords: []", text)
        self.assertNotIn("persons: []", text)
        self.assertNotIn("title: null", text)
        self.assertIn("date: '2026-05-29'", text)
        self.assertTrue(text.endswith("caption\n"))


if __name__ == "__main__":
    unittest.main()
