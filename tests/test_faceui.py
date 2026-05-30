import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient


class FaceUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        import faces

        self._faces = faces
        self._patch_root = patch.object(faces, "root", return_value=self.root)
        self._patch_root.start()
        self._faces.db()

        img = np.zeros((80, 100, 3), dtype="uint8")
        img[20:60, 30:70] = 255
        face1 = faces.FaceEmbedding(np.array([1, 0], dtype="float32").tobytes(), (30, 20, 70, 60), 0.95)
        face2 = faces.FaceEmbedding(np.array([0, 1], dtype="float32").tobytes(), (10, 10, 50, 50), 0.91)
        self.id1 = faces.store_face_embeddings("onedrive", "/tmp/a.jpg", [face1], img)[0]
        self.id2 = faces.store_face_embeddings("onedrive", "/tmp/b.jpg", [face2], img)[0]
        faces.label_face(self.id1, "Tejas")

        from faceui import app
        self.client = TestClient(app)

    def tearDown(self):
        self._patch_root.stop()
        self.tmp.cleanup()

    def test_grid_lists_only_unlabelled_faces(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        body = response.text
        self.assertIn(f'name="face_{self.id2}"', body)
        self.assertNotIn(f'name="face_{self.id1}"', body)

    def test_grid_paginates(self):
        self._faces.label_face(self.id2, "Ishwa")
        response = self.client.get("/?page=1&size=5")
        self.assertEqual(200, response.status_code)
        self.assertIn("No unlabelled faces", response.text)

    def test_label_endpoint_batch_persists(self):
        response = self.client.post("/label", data={
            f"face_{self.id2}": "Ishwa",
        })
        self.assertEqual(200, response.status_code)
        name = self._faces.name_for_face(self.id2)
        self.assertEqual("Ishwa", name)

    def test_label_endpoint_skips_blanks(self):
        self._faces.label_face(self.id2, "Ishwa")
        response = self.client.post("/label", data={
            f"face_{self.id2}": "",
        })
        self.assertEqual(200, response.status_code)
        name = self._faces.name_for_face(self.id2)
        self.assertEqual("Ishwa", name)

    def test_serve_crop_returns_jpeg_and_404s_missing(self):
        response = self.client.get(f"/faces/{self.id1}.jpg")
        self.assertEqual(200, response.status_code)
        self.assertEqual("image/jpeg", response.headers["content-type"])

        response = self.client.get("/faces/99999.jpg")
        self.assertEqual(404, response.status_code)

    def test_names_returns_distinct_existing_names(self):
        self._faces.label_face(self.id2, "Ishwa")
        response = self.client.get("/names")
        self.assertEqual(200, response.status_code)
        data = response.json()
        self.assertEqual(["Ishwa", "Tejas"], sorted(data["names"]))


if __name__ == "__main__":
    unittest.main()
