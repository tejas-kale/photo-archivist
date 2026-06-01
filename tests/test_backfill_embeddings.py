import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class BackfillEmbeddingsTests(unittest.TestCase):
    def test_backfill_embeddings_updates_missing_rows(self):
        import archive

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "archive.db"
            image = Path(d) / "a.jpg"
            image.write_bytes(b"jpg")
            con = sqlite3.connect(db_path)
            con.execute("create table media (id text primary key, original_path text, embedding blob)")
            con.execute("insert into media values ('a', ?, null)", (str(image),))
            con.commit()
            with patch.object(archive.embed, "embedding_blob_subprocess", return_value=b"vector") as embed:
                created, skipped = archive.backfill_embeddings(db_path, None)
            blob = con.execute("select embedding from media where id = 'a'").fetchone()[0]

        self.assertEqual((1, 0), (created, skipped))
        self.assertEqual(b"vector", blob)
        embed.assert_called_once_with(image)

    def test_backfill_embeddings_skips_failures_and_honours_limit(self):
        import archive

        with tempfile.TemporaryDirectory() as d:
            db_path = Path(d) / "archive.db"
            image1 = Path(d) / "a.jpg"
            image2 = Path(d) / "b.jpg"
            image1.write_bytes(b"jpg")
            image2.write_bytes(b"jpg")
            con = sqlite3.connect(db_path)
            con.execute("create table media (id text primary key, original_path text, embedding blob)")
            con.execute("insert into media values ('a', ?, null)", (str(image1),))
            con.execute("insert into media values ('b', ?, null)", (str(image2),))
            con.commit()
            with patch.object(archive.embed, "embedding_blob_subprocess", side_effect=RuntimeError("missing torch")):
                created, skipped = archive.backfill_embeddings(db_path, 1)

        self.assertEqual((0, 1), (created, skipped))


if __name__ == "__main__":
    unittest.main()
