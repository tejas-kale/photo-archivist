import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


class SourceSelectionTests(unittest.TestCase):
    def test_media_can_return_latest_first_by_mtime(self):
        from photo_archivist.sources import onedrive

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root / "old.jpg"
            new = root / "new.jpg"
            old.write_bytes(b"jpg")
            new.write_bytes(b"jpg")
            old_time = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
            new_time = datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp()
            old.touch()
            new.touch()
            import os
            os.utime(old, (old_time, old_time))
            os.utime(new, (new_time, new_time))

            found = list(onedrive.media(root, limit=1, selection="latest"))

        self.assertEqual([new.resolve()], [item.path for item in found])

    def test_media_filters_by_file_mtime_period_before_sampling(self):
        from photo_archivist.sources import onedrive

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            keep = root / "keep.jpg"
            skip = root / "skip.jpg"
            keep.write_bytes(b"jpg")
            skip.write_bytes(b"jpg")
            import os
            os.utime(keep, (datetime(2026, 1, 5, tzinfo=timezone.utc).timestamp(),) * 2)
            os.utime(skip, (datetime(2026, 1, 9, tzinfo=timezone.utc).timestamp(),) * 2)
            with patch.object(onedrive.random, "sample", return_value=[keep.resolve()]) as sample:
                found = list(onedrive.media(root, limit=1, start=datetime(2026, 1, 1, tzinfo=timezone.utc), end=datetime(2026, 1, 6, tzinfo=timezone.utc)))

        sample.assert_not_called()
        self.assertEqual([keep.resolve()], [item.path for item in found])


if __name__ == "__main__":
    unittest.main()
