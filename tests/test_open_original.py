import unittest
from pathlib import Path
from unittest.mock import patch


class OpenOriginalTests(unittest.TestCase):
    def test_open_original_uses_open_reveal(self):
        import open_original

        with patch.object(open_original.subprocess, "run") as run:
            open_original.open_original(Path("/tmp/x.jpg"))

        run.assert_called_once_with(["open", "-R", "/tmp/x.jpg"], check=True)


if __name__ == "__main__":
    unittest.main()
