import tempfile
import unittest
from pathlib import Path

from photo_archivist.describe import VisionResult
from photo_archivist.metadata import PhotoMetadata
from photo_archivist.sources.base import SourceMedia


class SidecarTests(unittest.TestCase):
    def test_sidecar_omits_location_and_faces_when_empty(self):
        from photo_archivist import sidecar

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            media = SourceMedia("onedrive", "id", image, {})
            path = sidecar.write(media, VisionResult(description_prose="caption"), PhotoMetadata(None, None, None, None, None, None, None, None, None))
            text = path.read_text()

        self.assertNotIn("location:", text)
        self.assertNotIn("faces:", text)
        self.assertTrue(text.endswith("## Description\ncaption\n"))


if __name__ == "__main__":
    unittest.main()
