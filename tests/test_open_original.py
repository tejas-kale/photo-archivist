import unittest
from unittest.mock import patch


class OpenOriginalTests(unittest.TestCase):
    def test_open_photos_uses_media_item_id(self):
        import open_original

        with patch.object(open_original.subprocess, "run") as run:
            open_original.open_original("photos", "uuid", None)

        run.assert_called_once_with(["osascript", "-e", 'tell application "Photos" to spotlight media item id "uuid"'], check=True)


if __name__ == "__main__":
    unittest.main()
