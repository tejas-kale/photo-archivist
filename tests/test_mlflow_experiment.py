import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class MlflowExperimentTests(unittest.TestCase):
    def test_processed_images_selects_processed_rows(self):
        import mlflow_experiment as exp

        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "archive.db"
            con = sqlite3.connect(db)
            con.execute("create table media (id text, original_path text, description text, indexed_at text)")
            con.execute("insert into media values ('a', 'a.jpg', 'desc', '2026')")
            con.execute("insert into media values ('b', 'b.jpg', null, '2026')")
            con.commit()
            rows = exp.processed_images(db, 50)

        self.assertEqual(1, len(rows))
        self.assertEqual("a", rows[0]["id"])

    def test_log_image_logs_original_existing_and_generated_description(self):
        import describe
        import mlflow_experiment as exp

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "a.jpg"
            image.write_bytes(b"image")
            image.with_name("a.description.md").write_text("existing")
            row = {"id": "a", "original_path": str(image), "description": "db desc", "indexed_at": "now"}
            result = describe.VisionResult(description_prose="new desc")
            with patch.object(exp.onedrive, "ensure_local", return_value=image), patch.object(exp.describe, "describe", return_value=result) as describe_image, patch.object(exp.mlflow, "log_artifacts") as log_artifacts, patch.object(exp.time, "monotonic", side_effect=[1, 3]):
                seconds = exp.log_image(row, "ollama", "model", root / "out")

            out = root / "out" / "a"
            self.assertEqual(2, seconds)
            describe_image.assert_called_once_with(image, backend="ollama", model="model", retries=0)
            self.assertEqual("existing", (out / "existing.description.md").read_text())
            self.assertIn("new desc", (out / "generated.description.md").read_text())
            self.assertTrue((out / "a.jpg").exists())
            log_artifacts.assert_called_once()


if __name__ == "__main__":
    unittest.main()
