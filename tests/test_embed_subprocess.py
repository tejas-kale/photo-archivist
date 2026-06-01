import base64
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class EmbedSubprocessTests(unittest.TestCase):
    def test_embedding_blob_subprocess_calls_module(self):
        import embed

        run = Mock(returncode=0, stdout=base64.b64encode(b"vector").decode(), stderr="")
        with patch.object(embed.subprocess, "run", return_value=run) as call:
            blob = embed.embedding_blob_subprocess(Path("x.jpg"))

        self.assertEqual(b"vector", blob)
        call.assert_called_once_with([sys.executable, "-m", "embed", "x.jpg"], check=False, capture_output=True, text=True)

    def test_embedding_blob_subprocess_includes_stderr_on_failure(self):
        import embed

        run = Mock(returncode=1, stdout="", stderr="missing torch")
        with patch.object(embed.subprocess, "run", return_value=run):
            with self.assertRaises(RuntimeError) as ctx:
                embed.embedding_blob_subprocess(Path("x.jpg"))

        self.assertIn("missing torch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
