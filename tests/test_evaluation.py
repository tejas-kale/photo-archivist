import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


class EvaluationTests(unittest.TestCase):
    def test_archive_events_emit_vision_output_for_pipeline_consumers(self):
        from photo_archivist import archive_runner, describe
        from photo_archivist.sources.base import SourceMedia

        media = SourceMedia("eval", "a", Path("/tmp/a.jpg"), {})
        metadata = Mock(gps_lat=None, gps_lon=None)
        result = describe.VisionResult(rating="keep", keywords=["dog"])
        options = archive_runner.ArchiveOptions(source="eval", db_path="archive.db", write_geocode=False, write_faces=False, write_sidecar=False)
        with patch.object(archive_runner.metadata, "extract_metadata", return_value=metadata), patch.object(archive_runner.describe, "describe", return_value=result), patch.object(archive_runner.store, "save"):
            events = list(archive_runner.archive_events(options, source_func=lambda *args: [media]))

        vision = [e for e in events if e["type"] == "vision"]
        self.assertEqual(1, len(vision))
        self.assertEqual("/tmp/a.jpg", vision[0]["path"])
        self.assertEqual("keep", vision[0]["data"].rating)

    def test_query_candidates_uses_archive_descriptions_and_fields(self):
        from photo_archivist import evaluation
        import sqlite3

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "archive.db"
            outdoor = root / "outdoor.jpg"
            people = root / "people.jpg"
            blurry = root / "blurry.jpg"
            for image in [outdoor, people, blurry]:
                image.write_bytes(b"jpg")
            con = sqlite3.connect(db)
            con.execute("create table media (id text primary key, source text, original_path text, description text, number_people integer, day_night text, lighting_quality text, blur integer, picture_quality text, face_count integer, indexed_at text)")
            con.execute("insert into media values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("outdoor", "onedrive", str(outdoor), "Sunny park with blue sky and trees", 0, "day", "bright", 0, "good", 0, "2026"))
            con.execute("insert into media values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("people", "onedrive", str(people), "Family portrait with several people", 3, "day", "soft", 0, "good", 2, "2026"))
            con.execute("insert into media values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("blurry", "onedrive", str(blurry), "Dark blurry underexposed photo", 0, "night", "poor", 1, "poor", 0, "2026"))
            con.commit()
            pool = evaluation.query_candidates(db, root / "eval")

        self.assertIn(str(outdoor), [c["path"] for c in pool["categories"]["well-lit-outdoor"]])
        self.assertIn(str(people), [c["path"] for c in pool["categories"]["people"]])
        self.assertIn(str(blurry), [c["path"] for c in pool["categories"]["bad-exposure-blurry"]])

    def test_classify_image_uses_fast_ollama_category_prompt(self):
        from photo_archivist import evaluation

        response = Mock()
        response.json.return_value = {"response": "people"}
        with patch.object(evaluation.describe, "image_data", return_value="image"), patch.object(evaluation.httpx, "post", return_value=response) as post:
            category = evaluation.classify_image(Path("x.jpg"), model="gemma4:e2b")

        body = post.call_args.kwargs["json"]
        self.assertEqual("people", category)
        self.assertEqual("gemma4:e2b", body["model"])
        self.assertEqual(["image"], body["images"])
        self.assertEqual(16, body["options"]["num_predict"])
        self.assertIn("well-lit-outdoor", body["prompt"])
        self.assertNotIn("description_prose", body["prompt"])

    def test_classify_candidates_merges_until_target(self):
        from photo_archivist import evaluation
        import sqlite3

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            db = root / "archive.db"
            image = root / "pet.jpg"
            image.write_bytes(b"jpg")
            con = sqlite3.connect(db)
            con.execute("create table media (id text primary key, source text, original_path text, description text)")
            con.execute("insert into media values (?, ?, ?, ?)", ("pet", "onedrive", str(image), ""))
            con.commit()
            with patch.object(evaluation, "classify_image", return_value="unusual-subjects"):
                pool = evaluation.classify_candidates(db, root / "eval", sample_size=1, target_per_category=1)

        self.assertIn(str(image), [c["path"] for c in pool["categories"]["unusual-subjects"]])

    def test_export_candidates_writes_resized_jpegs_and_manifest(self):
        from photo_archivist import evaluation

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "source.jpg"
            Image.new("RGB", (200, 100), "red").save(image, exif=b"Exif\x00\x00fake")
            pool = {"categories": {"people": [{"id": "c1", "path": str(image), "category": "people", "source": "db"}]}, "skipped": []}
            evaluation.save_pool(root / "eval", pool)

            manifest = evaluation.export_candidates(root / "eval", root / "upload", limit_per_category=1, max_size=64, quality=80)

            out = root / "upload" / "people" / "c1.jpg"
            self.assertEqual([{"id": "c1", "category": "people", "file": "people/c1.jpg", "original_path": str(image)}], manifest)
            self.assertTrue(out.exists())
            with Image.open(out) as exported:
                self.assertLessEqual(max(exported.size), 64)
                self.assertFalse(exported.getexif())
            self.assertEqual(manifest, json.loads((root / "upload" / "manifest.json").read_text()))

    def test_import_drafts_and_next_candidate_exposes_prefill(self):
        from photo_archivist import evaluation

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "candidate.jpg"
            image.write_bytes(b"jpg")
            pool = {"categories": {"people": [{"id": "c1", "path": str(image), "category": "people", "source": "db"}]}, "skipped": []}
            evaluation.save_pool(root / "eval", pool)
            incoming = root / "out" / "people"
            incoming.mkdir(parents=True)
            (incoming / "c1.json").write_text(json.dumps({"rating": "keep", "keywords": ["family"], "description_prose": "A family photo.", "source": "qwen-draft"}))

            imported = evaluation.import_drafts(root / "eval", root / "out")
            candidate = evaluation.next_candidate(root / "eval", "people")["candidate"]

        self.assertEqual(1, imported)
        self.assertEqual("keep", candidate["draft"]["rating"])
        self.assertEqual(["family"], candidate["draft"]["keywords"])

    def test_label_candidate_preserves_draft_fields_but_marks_reviewed(self):
        from photo_archivist import evaluation

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "candidate.jpg"
            image.write_bytes(b"jpg")
            pool = {"categories": {"people": [{"id": "c1", "path": str(image), "category": "people", "source": "db"}]}, "skipped": []}
            evaluation.save_pool(root / "eval", pool)
            draft_dir = root / "eval" / "drafts" / "people"
            draft_dir.mkdir(parents=True)
            (draft_dir / "c1.json").write_text(json.dumps({"focus": "sharp", "source": "qwen-draft", "rating": "review", "keywords": ["old"], "description_prose": "Old."}))
            with patch.object(evaluation.onedrive, "ensure_local", return_value=image):
                data = evaluation.label_candidate(root / "eval", "people", "c1", "keep", "family, portrait", "A family portrait.")

        self.assertEqual("sharp", data["focus"])
        self.assertEqual("human-reviewed", data["source"])
        self.assertEqual("qwen-draft", data["draft_source"])
        self.assertEqual(["family", "portrait"], data["keywords"])

    def test_score_runs_archive_pipeline_logs_mlflow_and_returns_metrics(self):
        from photo_archivist import describe, evaluation

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "images" / "well-lit-outdoor" / "a.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"jpg")
            golden = root / "golden"
            golden.mkdir()
            (golden / "a.json").write_text(json.dumps({"rating": "keep", "keywords": ["dog", "grass", "person"], "description_prose": "A dog on grass."}))
            result = describe.VisionResult(rating="keep", keywords=["dog", "park"], description_prose="A dog in a park.", people_count=None)
            events = [{"type": "total", "total": 1}, {"type": "vision", "path": str(image), "data": result}, {"type": "done", "processed": 1, "attempted": 1}]
            with patch.object(evaluation.archive_runner, "archive_events", return_value=iter(events)) as archive_events, patch.object(evaluation.mlflow, "set_tracking_uri") as tracking, patch.object(evaluation.mlflow, "set_experiment") as experiment, patch.object(evaluation.mlflow, "start_run", return_value=nullcontext()), patch.object(evaluation.mlflow, "log_params") as params, patch.object(evaluation.mlflow, "log_metrics") as metrics, patch.object(evaluation.mlflow, "log_artifacts") as artefacts:
                summary = evaluation.score(root, tracking_uri="sqlite:///mlflow.db", experiment="eval")

        archive_events.assert_called_once()
        tracking.assert_called_once_with("sqlite:///mlflow.db")
        experiment.assert_called_once_with("eval")
        params.assert_called_once()
        logged = metrics.call_args.args[0]
        self.assertEqual(1, summary["images_scored"])
        self.assertEqual(1.0, summary["rating_accuracy"])
        self.assertEqual(1 / 3, summary["keyword_coverage"])
        self.assertEqual(summary["keyword_coverage"], logged["keyword_coverage"])
        self.assertEqual(0, logged["images_missing_goldens"])
        artefacts.assert_called_once()

    def test_cli_eval_candidate_subcommands(self):
        from click.testing import CliRunner
        from photo_archivist import cli, evaluation

        runner = CliRunner()
        with patch.object(evaluation, "query_candidates", return_value={"categories": {"people": [{"id": "a"}]}}) as query:
            result = runner.invoke(cli.cli, ["eval", "query-candidates", "--db", "archive.db", "--eval-dir", "eval"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("people", result.output)
        query.assert_called_once()

        with patch.object(evaluation, "classify_candidates", return_value={"categories": {"people": [{"id": "a"}]}}) as classify:
            result = runner.invoke(cli.cli, ["eval", "classify-candidates", "--db", "archive.db", "--eval-dir", "eval", "--sample-size", "1"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("people", result.output)
        classify.assert_called_once()

        with patch.object(evaluation, "export_candidates", return_value=[{"id": "a"}, {"id": "b"}]) as export:
            result = runner.invoke(cli.cli, ["eval", "export-candidates", "--eval-dir", "eval", "--output", "eval/upload"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("exported 2 candidates", result.output)
        export.assert_called_once()

        with patch.object(evaluation, "import_drafts", return_value=2) as import_drafts:
            result = runner.invoke(cli.cli, ["eval", "import-drafts", "--eval-dir", "eval", "--draft-dir", "eval_out"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("imported 2 drafts", result.output)
        import_drafts.assert_called_once()

    def test_cli_eval_score_subcommand(self):
        from click.testing import CliRunner
        from photo_archivist import cli, evaluation

        runner = CliRunner()
        with patch.object(evaluation, "score", return_value={"images_scored": 1, "images_missing_goldens": 0, "images_failed": 0, "rating_accuracy": 1.0, "keyword_coverage": 0.5, "field_completeness": 1.0}) as score:
            result = runner.invoke(cli.cli, ["eval", "score", "--eval-dir", "eval"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("rating_accuracy", result.output)
        score.assert_called_once()


if __name__ == "__main__":
    unittest.main()
