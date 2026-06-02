import sqlite3
import tempfile
import unittest
from pathlib import Path


class SearchTests(unittest.TestCase):
    def test_search_matches_description_and_sidecar(self):
        import search

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db_path = root / "archive.db"
            beach = root / "beach.jpg"
            forest = root / "forest.jpg"
            beach.write_bytes(b"jpg")
            forest.write_bytes(b"jpg")
            forest.with_name("forest.description.md").write_text("misty hill walk")
            con = sqlite3.connect(db_path)
            con.execute("create table media (id text primary key, original_path text, description text, activity text, place text, indexed_at text)")
            con.execute("insert into media values (?, ?, ?, ?, ?, ?)", ("1", str(beach), "sunset beach", "swimming", "Goa", "2026-01-02"))
            con.execute("insert into media values (?, ?, ?, ?, ?, ?)", ("2", str(forest), "trees", "walking", "Wales", "2026-01-01"))
            con.commit()

            text = search.find(db_path, "beach")
            sidecar = search.find(db_path, "misty")

        self.assertEqual(["1"], [row["id"] for row in text])
        self.assertEqual(["2"], [row["id"] for row in sidecar])
        self.assertEqual("misty hill walk", sidecar[0]["text"])

    def test_search_empty_query_returns_empty(self):
        import search

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual([], search.find(Path(d) / "missing.db", ""))


if __name__ == "__main__":
    unittest.main()
