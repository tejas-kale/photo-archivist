import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient


class WebUITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.db_path = self.root / "archive.db"
        self.image = self.root / "a.jpg"
        self.image.write_bytes(b"jpg")
        con = sqlite3.connect(self.db_path)
        con.execute("create table media (id text primary key, original_path text, description text, activity text, place text, indexed_at text)")
        con.execute("insert into media values (?, ?, ?, ?, ?, ?)", ("1", str(self.image), "red kite", "flying", "park", "2026"))
        con.commit()
        from photo_archivist.web import app as webui
        webui.DB_PATH = self.db_path
        webui.reset_job()
        self.webui = webui
        self.client = TestClient(webui.app)

    def tearDown(self):
        self.tmp.cleanup()

    def test_homepage_has_three_tabs(self):
        response = self.client.get("/")
        self.assertEqual(200, response.status_code)
        self.assertIn("Photo Archiver", response.text)
        self.assertIn("Archive", response.text)
        self.assertIn("Faces", response.text)
        self.assertIn("Search", response.text)
        self.assertIn('id="sourceChoice"', response.text)
        self.assertIn('id="model"', response.text)
        self.assertIn('id="startWrap" class="field hidden"', response.text)
        self.assertIn(r"j.logs.join('\n')", response.text)
        self.assertIn("grid-template-columns:220px 120px 210px 210px 140px", response.text)
        self.assertIn("width:80vw;margin:0 auto", response.text)
        self.assertIn('<div class="toolbar"><button id="loadFaces" class="primary">Refresh</button><button id="saveFaces" class="primary">Save</button></div>', response.text)

    def test_search_endpoint_uses_shared_search(self):
        response = self.client.get("/api/search?q=kite")
        self.assertEqual(200, response.status_code)
        self.assertEqual("1", response.json()["results"][0]["id"])

    def test_image_route_ensures_local(self):
        with patch.object(self.webui.onedrive, "ensure_local", return_value=self.image) as ensure:
            response = self.client.get("/api/images/1")
        self.assertEqual(200, response.status_code)
        ensure.assert_called_once_with(self.image)

    def test_archive_job_runs_shared_backend(self):
        from photo_archivist.sources.base import SourceMedia
        item = SourceMedia("onedrive", "id", Path("x.jpg"), {})
        data = self.webui.describe.VisionResult(description_prose="caption")
        with patch.object(self.webui.archive_runner, "source_media", return_value=[item]), patch.object(self.webui.metadata, "extract_metadata", return_value=Mock(gps_lat=None, gps_lon=None)), patch.object(self.webui.describe, "describe", return_value=data), patch.object(self.webui.archive_runner.embed, "embedding_blob_subprocess", return_value=b"vector") as embed, patch.object(self.webui.faces, "detect_faces", return_value=([], None)), patch.object(self.webui.faces, "store_face_embeddings", return_value=[]), patch.object(self.webui.store, "save"), patch.object(self.webui.sidecars, "write"), patch.object(self.webui.archive_runner.ollama_ctl, "restart") as restart, patch.object(self.webui.archive_runner.ollama_ctl, "stop") as stop:
            response = self.client.post("/api/archive/start", json={"limit": 1, "model": "gemma4:e2b", "selection": "latest"})
            self.assertEqual(200, response.status_code)
            for _ in range(50):
                status = self.client.get("/api/archive/status").json()
                if status["status"] == "done":
                    break
                time.sleep(0.02)

        self.assertEqual("done", status["status"])
        self.assertEqual(1, status["total"])
        self.assertEqual(1, status["processed"])
        embed.assert_called_once_with(item.path)
        restart.assert_called_once_with(5)
        stop.assert_called_once_with()
        self.assertIn("🦙 restarting Ollama", "\n".join(status["logs"]))
        self.assertIn("🧬 embedding", "\n".join(status["logs"]))
        self.assertIn("✅ archived", "\n".join(status["logs"]))


if __name__ == "__main__":
    unittest.main()
