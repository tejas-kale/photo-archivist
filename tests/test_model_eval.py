import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ModelEvalTests(unittest.TestCase):
    def test_images_reads_non_empty_lines(self):
        import model_eval

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "images.txt"
            path.write_text("a.jpg\n\n# no\nb.heic\n")
            found = model_eval.images(path)

        self.assertEqual([Path("a.jpg"), Path("b.heic")], found)

    def test_run_writes_blind_outputs_key_and_feedback_template(self):
        import describe
        import model_eval

        with tempfile.TemporaryDirectory() as d:
            image_list = Path(d) / "images.txt"
            image_list.write_text("a.jpg\n")
            out = Path(d) / "results"
            result = describe.VisionResult(rating="keep", people_count=2, description_prose="caption")
            with patch.object(model_eval.describe, "describe", return_value=result) as describe_image, patch.object(model_eval.time, "monotonic", side_effect=[1, 3, 4, 9]), patch.object(model_eval.random, "shuffle", side_effect=lambda x: None):
                rows = model_eval.run(image_list, ["m1", "m2"], out, retries=0)

            with out.with_name("results_blind.csv").open() as f:
                blind_rows = list(csv.DictReader(f))
            with out.with_name("results_key.csv").open() as f:
                key_rows = list(csv.DictReader(f))
            with out.with_name("results_feedback_template.csv").open() as f:
                feedback_rows = list(csv.DictReader(f))
            jsonl = [json.loads(line) for line in out.with_name("results_blind.jsonl").read_text().splitlines()]

        self.assertEqual(2, len(rows))
        self.assertEqual(2, describe_image.call_count)
        self.assertEqual(["A", "B"], [r["variant"] for r in blind_rows])
        self.assertNotIn("model", blind_rows[0])
        self.assertEqual(["m1", "m2"], [r["model"] for r in key_rows])
        self.assertEqual(["A", "B"], [r["variant"] for r in feedback_rows])
        self.assertIn("description_score", feedback_rows[0])
        self.assertEqual("A", jsonl[0]["variant"])
        self.assertNotIn("model", jsonl[0])

    def test_blind_rows_assigns_letters_per_image(self):
        import model_eval

        rows = [{"image": "a", "model": "m1"}, {"image": "a", "model": "m2"}, {"image": "b", "model": "m1"}]
        with patch.object(model_eval.random, "shuffle", side_effect=lambda x: x.reverse()):
            blind, key = model_eval.blind_rows(rows)

        self.assertEqual(["A", "B", "A"], [r["variant"] for r in blind])
        self.assertEqual(["m2", "m1", "m1"], [r["model"] for r in key])
        self.assertNotIn("model", blind[0])

    def test_evaluate_records_errors(self):
        import model_eval

        with patch.object(model_eval.describe, "describe", side_effect=RuntimeError("bad")), patch.object(model_eval.time, "monotonic", side_effect=[1, 2]):
            row = model_eval.evaluate(Path("x.jpg"), "m", retries=0)

        self.assertFalse(row["ok"])
        self.assertEqual("bad", row["error"])


if __name__ == "__main__":
    unittest.main()
