import base64
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class EmbedSubprocessTests(unittest.TestCase):
    def test_embedding_blob_subprocess_calls_module(self):
        import embed

        run = Mock(stdout=base64.b64encode(b"vector").decode())
        with patch.object(embed.subprocess, "run", return_value=run) as call:
            blob = embed.embedding_blob_subprocess(Path("x.jpg"))

        self.assertEqual(b"vector", blob)
        call.assert_called_once_with([sys.executable, "-m", "embed", "x.jpg"], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
