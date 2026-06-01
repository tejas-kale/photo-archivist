import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class ReviewUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "archive.db"
        self.image1 = self.root / "a.jpg"
        self.image2 = self.root / "b.jpg"
        self.image3 = self.root / "c.jpg"
        self.image4 = self.root / "d.jpg"
        for image in [self.image1, self.image2, self.image3, self.image4]:
            image.write_bytes(b"jpg")
        con = sqlite3.connect(self.db_path)
        con.execute("create table media (id text primary key, original_path text, description text, rating text, people_count int, indexed_at text)")
        for i, image in enumerate([self.image1, self.image2, self.image3, self.image4], 1):
            con.execute("insert into media values (?, ?, ?, ?, ?, ?)", (str(i), str(image), f"description {i}", "keep", i, f"2026-05-31T00:0{i}:00+00:00"))
        con.commit()
        import reviewui
        reviewui.DB_PATH = self.db_path
        self.client = TestClient(reviewui.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_grid_shows_three_images_and_descriptions(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn("description 4", response.text)
        self.assertIn("description 2", response.text)
        self.assertNotIn("description 1", response.text)
        self.assertEqual(3, response.text.count('<img src="/images/'))

    def test_page_two_shows_remaining_image(self):
        response = self.client.get("/?page=2")
        self.assertEqual(200, response.status_code)
        self.assertIn("description 1", response.text)
        self.assertEqual(1, response.text.count('<img src="/images/'))

    def test_image_route_ensures_local_and_returns_file(self):
        import reviewui

        with patch.object(reviewui.onedrive, "ensure_local", return_value=self.image1) as ensure:
            response = self.client.get("/images/1")

        self.assertEqual(200, response.status_code)
        ensure.assert_called_once_with(self.image1)
        self.assertEqual(b"jpg", response.content)

    def test_size_is_capped_at_three(self):
        response = self.client.get("/?size=99")
        self.assertEqual(200, response.status_code)
        self.assertEqual(3, response.text.count('<img src="/images/'))

    def test_grid_handles_legacy_schema_without_rating(self):
        import reviewui

        db_path = self.root / "legacy.db"
        con = sqlite3.connect(db_path)
        con.execute("create table media (id text primary key, original_path text, description text, number_people int, indexed_at text)")
        con.execute("insert into media values (?, ?, ?, ?, ?)", ("legacy", str(self.image1), "legacy description", 2, "2026-05-31T00:00:00+00:00"))
        con.commit()
        reviewui.DB_PATH = db_path

        response = self.client.get("/")

        self.assertEqual(200, response.status_code)
        self.assertIn("legacy description", response.text)
        self.assertIn("People", response.text)


if __name__ == "__main__":
    unittest.main()
