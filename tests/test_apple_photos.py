import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sources import apple_photos


def photo(path, uuid="uuid", filename="image.jpg"):
    return SimpleNamespace(
        path=str(path) if path is not None else None,
        path_derivatives=[],
        uuid=uuid,
        filename=filename,
        date=None,
        title=None,
        description=None,
        keywords=[],
        albums=[],
        persons=[],
    )


def test_is_stub_none_path():
    assert apple_photos.is_stub(photo(None)) is True


def test_is_stub_zero_size(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"")

    assert apple_photos.is_stub(photo(path)) is True


def test_is_stub_real_file(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")

    assert apple_photos.is_stub(photo(path)) is False


@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_already_present(run, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")

    assert apple_photos.ensure_local(photo(path)) == path
    run.assert_not_called()


@patch("sources.apple_photos.time.sleep")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_downloads_and_resolves(run, sleep, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    item = photo(path)

    with patch("sources.apple_photos.is_stub", side_effect=[True, True, False]):
        assert apple_photos.ensure_local(item, timeout_s=5) == path

    run.assert_called_once_with(["brctl", "download", str(path)], check=True, capture_output=True)


@patch("sources.apple_photos.time.sleep")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_downloads_pathless_photo(run, sleep, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    item = photo(None)
    item._path_5 = Mock(return_value=str(path))

    with patch("sources.apple_photos.is_stub", side_effect=[True, False]):
        assert apple_photos.ensure_local(item, timeout_s=5) == path

    run.assert_called_once_with(["brctl", "download", str(path)], check=True, capture_output=True)


@patch("sources.apple_photos.PhotoExporter")
@patch("sources.apple_photos.tempfile.mkdtemp")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_falls_back_to_photokit_export(run, mkdtemp, exporter, tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    mkdtemp.return_value = str(export_dir)
    path = tmp_path / "missing.jpg"
    exported = export_dir / "image.jpg"
    exported.write_bytes(b"x")
    item = photo(path)
    result = SimpleNamespace(error=[], exported=[str(exported)])
    exporter.return_value.export.return_value = result
    run.side_effect = subprocess.CalledProcessError(1, "brctl", stderr=b"outside")

    assert apple_photos.ensure_local(item, timeout_s=5) == exported
    exporter.assert_called_once_with(item)
    _, kwargs = exporter.return_value.export.call_args
    assert kwargs["filename"] == "image.jpg"
    assert kwargs["options"].download_missing is True
    assert kwargs["options"].use_photokit is True


@patch("sources.apple_photos.PhotoExporter")
@patch("sources.apple_photos.tempfile.mkdtemp")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_skips_brctl_for_photos_library(run, mkdtemp, exporter, tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    mkdtemp.return_value = str(export_dir)
    path = tmp_path / "Photos Library.photoslibrary" / "originals" / "A" / "image.jpg"
    exported = export_dir / "image.jpg"
    exported.write_bytes(b"x")
    result = SimpleNamespace(error=[], exported=[str(exported)])
    exporter.return_value.export.return_value = result

    assert apple_photos.ensure_local(photo(path), timeout_s=5) == exported
    run.assert_not_called()


@patch("sources.apple_photos.time.sleep")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_timeout(run, sleep, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")

    with patch("sources.apple_photos.is_stub", return_value=True), patch(
        "sources.apple_photos.time.monotonic", side_effect=[0, 0, 2]
    ):
        with pytest.raises(TimeoutError):
            apple_photos.ensure_local(photo(path), timeout_s=1, poll_interval_s=0)


@patch("sources.apple_photos.time.sleep")
@patch("sources.apple_photos.subprocess.run")
def test_ensure_local_logs_download(run, sleep, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")

    with patch("sources.apple_photos.is_stub", side_effect=[True, True, False]), patch(
        "sources.apple_photos.time.monotonic", side_effect=[0, 0, 1]
    ):
        apple_photos.ensure_local(photo(path, filename="image.jpg"), timeout_s=5)

    assert "Downloaded image.jpg from iCloud via brctl (1.0s)" in caplog.text


@patch("sources.apple_photos.subprocess.run")
def test_evict_local_success(run, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")

    apple_photos.evict_local(path)

    run.assert_called_once_with(["brctl", "evict", str(path)], check=True, capture_output=True)


@patch("sources.apple_photos.subprocess.run")
def test_evict_local_failure_logs_not_raises(run, tmp_path, caplog):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    run.side_effect = subprocess.CalledProcessError(1, "brctl", stderr=b"busy")

    apple_photos.evict_local(path)

    assert "Could not evict" in caplog.text
    assert "busy" in caplog.text


@patch("sources.apple_photos.subprocess.run")
def test_evict_local_missing_path(run, tmp_path):
    apple_photos.evict_local(tmp_path / "missing.jpg")

    run.assert_not_called()


@patch("sources.apple_photos.subprocess.run")
def test_evict_local_skips_photos_library(run, tmp_path):
    path = tmp_path / "Photos Library.photoslibrary" / "originals" / "A" / "image.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")

    apple_photos.evict_local(path)

    run.assert_not_called()


@patch("sources.apple_photos.evict_local")
def test_evict_already_indexed(evict, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    db = Mock()
    db.conn.execute.return_value = [(0, "source_type")]
    db.query.return_value = [{"source_id": "uuid"}]
    photos_db = Mock()
    photos_db.get_photo.return_value = photo(path)

    apple_photos.evict_already_indexed(db, photos_db)

    db.query.assert_called_once()
    photos_db.get_photo.assert_called_once_with("uuid")
    evict.assert_called_once_with(path)


@patch("sources.apple_photos.ensure_local")
def test_iter_photos_limit_counts_attempts(ensure):
    ensure.side_effect = TimeoutError("missing")

    assert list(apple_photos.iter_photos([photo(None, uuid="1"), photo(None, uuid="2")], limit=1)) == []
    assert ensure.call_count == 1


@patch("sources.apple_photos.ensure_local")
def test_iter_photos_stops_after_consecutive_failures(ensure, caplog):
    caplog.set_level(logging.WARNING)
    ensure.side_effect = TimeoutError("missing")

    list(apple_photos.iter_photos([photo(None, uuid="1"), photo(None, uuid="2"), photo(None, uuid="3")], max_consecutive_download_failures=2))

    assert ensure.call_count == 2
    assert "Stopped Apple Photos import after 2 consecutive iCloud download failures" in caplog.text


@patch("sources.apple_photos.ensure_local")
@patch("sources.apple_photos.release_local")
def test_iter_photos_evicts_after_yield(release, ensure, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    item = photo(path)
    ensure.return_value = path

    media = list(apple_photos.iter_photos([item], evict_after=True))

    assert media[0].path == path
    release.assert_called_once_with(path)


@patch("sources.apple_photos.ensure_local")
@patch("sources.apple_photos.release_local")
def test_iter_photos_evicts_on_exception(release, ensure, tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"x")
    ensure.return_value = path

    with pytest.raises(RuntimeError):
        for _ in apple_photos.iter_photos([photo(path)], evict_after=True):
            raise RuntimeError("stop")

    release.assert_called_once_with(path)
