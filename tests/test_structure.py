import importlib.metadata
import tempfile
import unittest
from httpx import HTTPError
from pathlib import Path
from unittest.mock import Mock, ANY, patch

from click.testing import CliRunner


class StructureTests(unittest.TestCase):
    def test_console_script_points_to_click_archive_cli(self):
        scripts = importlib.metadata.entry_points(group="console_scripts")
        script = next(s for s in scripts if s.name == "photo-archivist")
        self.assertEqual("archive:cli", script.value)

        import archive

        result = CliRunner().invoke(archive.cli, ["--help"])
        self.assertEqual(0, result.exit_code)
        self.assertIn("--source", result.output)

    def test_source_media_rejects_photos(self):
        import archive

        with self.assertRaises(ValueError) as ctx:
            list(archive.source_media("photos"))
        self.assertIn("Apple Photos", str(ctx.exception))

    def test_serve_faces_cli(self):
        import archive

        with patch("uvicorn.run") as run:
            result = CliRunner().invoke(archive.cli, ["serve-faces", "--host", "0.0.0.0", "--port", "9000"])
        self.assertEqual(0, result.exit_code)
        run.assert_called_once()
        self.assertEqual("0.0.0.0", run.call_args.kwargs["host"])
        self.assertEqual(9000, run.call_args.kwargs["port"])

    def test_label_face_cli(self):
        import archive

        with patch.object(archive.faces, "label_face") as label:
            result = CliRunner().invoke(archive.cli, ["label-face", "42", "Tejas"])

        self.assertEqual(0, result.exit_code)
        label.assert_called_once_with(42, "Tejas")
        self.assertIn("labelled face 42 as Tejas", result.output)

    def test_train_faces_cli(self):
        import archive

        with patch.object(archive.faces, "train_faces") as train:
            result = CliRunner().invoke(archive.cli, ["train-faces"])

        self.assertEqual(0, result.exit_code)
        train.assert_called_once_with()
        self.assertIn("classifier trained", result.output)

    def test_source_media_shape(self):
        from sources.base import SourceMedia

        media = SourceMedia("onedrive", "id", Path("x.jpg"), {"name": "x"})
        self.assertEqual("onedrive", media.source)
        self.assertEqual("id", media.source_id)
        self.assertEqual(Path("x.jpg"), media.path)
        self.assertEqual({"name": "x"}, media.metadata)

    def test_filesystem_source_finds_images(self):
        from sources.onedrive import media

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "a.jpg"
            text = root / "a.txt"
            image.write_bytes(b"jpg")
            text.write_text("no")

            found = list(media(root))

        self.assertEqual(1, len(found))
        self.assertEqual("onedrive", found[0].source)
        self.assertEqual(image.resolve(), found[0].path)

    def test_filesystem_source_finds_nested_images(self):
        from sources.onedrive import media

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = root / "a" / "b" / "c.heic"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"heic")

            found = list(media(root))

        self.assertEqual([image.resolve()], [item.path for item in found])

    def test_archive_cli_uses_onedrive_personal_pictures(self):
        import archive

        with patch.object(archive.onedrive, "media", return_value=[]) as media:
            list(archive.source_media("onedrive"))

        media.assert_called_once_with(Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "tejas" / "Pictures")

    def test_source_media_defaults_to_onedrive(self):
        import archive

        with patch.object(archive.onedrive, "media", return_value=[]) as media:
            list(archive.source_media())

        media.assert_called_once_with(Path.home() / "Library" / "CloudStorage" / "OneDrive-Personal" / "tejas" / "Pictures")

    def test_archive_cli_accepts_specific_image_path(self):
        import archive

        with tempfile.TemporaryDirectory() as d:
            image = Path(d) / "IMG_1234.jpeg"
            image.write_bytes(b"image")
            found = list(archive.source_media(None, image))

        self.assertEqual(1, len(found))
        self.assertEqual("onedrive", found[0].source)
        self.assertEqual(image.resolve(), found[0].path)

    def test_archive_cli_wires_description_embedding_store_and_sidecar(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {"k": "v"})
        data = archive.describe.VisionResult(description_prose="line 1\nline 2", activity="playing chess")
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(archive.describe, "describe", return_value=data) as describe, patch.object(archive.embed, "embedding_blob", return_value=b"vector") as embed, patch.object(archive.faces, "detect_faces", return_value=([], None)) as detect_faces, patch.object(archive.faces, "store_face_embeddings", return_value=[]) as store_faces, patch.object(archive.store, "save") as save, patch.object(archive.sidecars, "write", return_value=Path("x.jpg.description.md")) as sidecar:
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--db", "test.db"])

        self.assertEqual(0, result.exit_code)
        describe.assert_called_once_with(item.path, backend="ollama", model=None, retries=2)
        embed.assert_called_once_with(item.path)
        detect_faces.assert_called_once_with(item.path)
        store_faces.assert_called_once_with(item.source, item.source_id, [], None)
        save.assert_called_once_with(item, data, b"vector", "test.db", ANY, None, 0)
        sidecar.assert_called_once_with(item, data, ANY, None, [], [])

    def test_archive_cli_limit_does_not_request_extra_media(self):
        import archive
        from sources.base import SourceMedia

        def media():
            yield SourceMedia("onedrive", "id", Path("x.jpg"), {})
            raise AssertionError("extra media requested")

        data = archive.describe.VisionResult(description_prose="caption")
        with patch.object(archive, "source_media", return_value=media()), patch.object(archive.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(archive.describe, "describe", return_value=data), patch.object(archive.embed, "embedding_blob"), patch.object(archive.faces, "detect_faces", return_value=([], None)), patch.object(archive.faces, "store_face_embeddings", return_value=[]), patch.object(archive.store, "save"), patch.object(archive.sidecars, "write"):
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--limit", "1", "--no-embed"])

        self.assertEqual(0, result.exit_code)

    def test_archive_cli_can_skip_embeddings(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        data = archive.describe.VisionResult(description_prose="caption")
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(archive.describe, "describe", return_value=data), patch.object(archive.embed, "embedding_blob") as embed, patch.object(archive.faces, "detect_faces", return_value=([], None)), patch.object(archive.faces, "store_face_embeddings", return_value=[]), patch.object(archive.store, "save") as save, patch.object(archive.sidecars, "write"):
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--no-embed"])

        self.assertEqual(0, result.exit_code)
        embed.assert_not_called()
        save.assert_called_once_with(item, data, None, "archive.db", ANY, None, 0)

    def test_archive_cli_can_skip_geocode_and_faces(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        photo_metadata = Mock(gps_lat=51.5, gps_lon=-0.1)
        data = archive.describe.VisionResult(description_prose="caption")
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.metadata, "extract_metadata", return_value=photo_metadata), patch.object(archive.geocode, "reverse_geocode") as reverse, patch.object(archive.describe, "describe", return_value=data), patch.object(archive.embed, "embedding_blob", return_value=b"vector"), patch.object(archive.faces, "detect_faces") as detect_faces, patch.object(archive.faces, "store_face_embeddings") as store_faces, patch.object(archive.store, "save"), patch.object(archive.sidecars, "write"):
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--no-geocode", "--no-faces"])

        self.assertEqual(0, result.exit_code)
        reverse.assert_not_called()
        detect_faces.assert_not_called()
        store_faces.assert_not_called()

    def test_archive_cli_can_preview_image(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(archive.describe, "describe", return_value=archive.describe.VisionResult(description_prose="caption")), patch.object(archive.embed, "embedding_blob", return_value=b"vector"), patch.object(archive.faces, "detect_faces", return_value=([], None)), patch.object(archive.faces, "store_face_embeddings", return_value=[]), patch.object(archive.store, "save"), patch.object(archive.sidecars, "write"), patch.object(archive.subprocess, "run") as run:
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--preview"])

        self.assertEqual(0, result.exit_code)
        run.assert_called_once_with(["open", "-a", "Preview", item.path], check=True)

    def test_archive_cli_verbose_logs_steps(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(archive.describe, "describe", return_value=archive.describe.VisionResult(description_prose="caption")), patch.object(archive.embed, "embedding_blob", return_value=b"vector"), patch.object(archive.faces, "detect_faces", return_value=([], None)), patch.object(archive.faces, "store_face_embeddings", return_value=[]), patch.object(archive.store, "save"), patch.object(archive.sidecars, "write"):
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--verbose"])

        self.assertEqual(0, result.exit_code)
        self.assertIn("🧠 describing", result.output)
        self.assertIn("🧬 embedding", result.output)
        self.assertIn("💾 saving", result.output)

    def test_describe_retries_empty_and_http_errors(self):
        import describe

        payload = '{"number_people": 0, "day_night": "day", "lighting_quality": "soft", "blur": false, "picture_quality": "good", "child": false, "description": "caption", "activity": "standing outside"}'
        with patch.object(describe, "describe_ollama", side_effect=["", HTTPError("down"), payload]) as ollama:
            text = describe.describe(Path("x.jpg"), backend="ollama", retries=2)

        self.assertEqual("caption", text["description"])
        self.assertEqual(3, ollama.call_count)


if __name__ == "__main__":
    unittest.main()
