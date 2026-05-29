import importlib.metadata
import tempfile
import unittest
from httpx import HTTPError
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_archive_cli_wires_description_embedding_store_and_sidecar(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {"k": "v"})
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.describe, "describe", return_value="caption") as describe, patch.object(archive.embed, "embedding_blob", return_value=b"vector") as embed, patch.object(archive.store, "save") as save, patch.object(archive.sidecars, "write", return_value=Path("x.jpg.description.md")) as sidecar:
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--db", "test.db"])

        self.assertEqual(0, result.exit_code)
        describe.assert_called_once_with(item.path, backend="ollama", model=None, retries=2)
        embed.assert_called_once_with(item.path)
        save.assert_called_once_with(item, "caption", b"vector", "test.db")
        sidecar.assert_called_once_with(item, "caption")

    def test_archive_cli_can_preview_image(self):
        import archive
        from sources.base import SourceMedia

        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        with patch.object(archive, "source_media", return_value=[item]), patch.object(archive.describe, "describe", return_value="caption"), patch.object(archive.embed, "embedding_blob", return_value=b"vector"), patch.object(archive.store, "save"), patch.object(archive.sidecars, "write"), patch.object(archive.subprocess, "run") as run:
            result = CliRunner().invoke(archive.cli, ["--source", "onedrive", "--preview"])

        self.assertEqual(0, result.exit_code)
        run.assert_called_once_with(["open", "-a", "Preview", item.path], check=True)

    def test_describe_retries_empty_and_http_errors(self):
        import describe

        with patch.object(describe, "describe_ollama", side_effect=["", HTTPError("down"), "caption"]) as ollama:
            text = describe.describe(Path("x.jpg"), backend="ollama", retries=2)

        self.assertEqual("caption", text)
        self.assertEqual(3, ollama.call_count)


if __name__ == "__main__":
    unittest.main()
