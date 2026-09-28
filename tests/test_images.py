from pathlib import Path

import pandas as pd
import pytest

from foodsearch.images import ImageIndex, media_type

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _manifest(path: Path, rows: list[tuple[str, int, str, str]]) -> Path:
    frame = pd.DataFrame(rows, columns=["itemId", "image_idx", "file", "status"])
    frame.to_csv(path, index=False)
    return path


def test_first_stored_image_wins_and_failures_are_skipped(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "b.jpg").write_bytes(JPEG)
    (images / "c.jpg").write_bytes(PNG)
    manifest = _manifest(
        tmp_path / "manifest.csv",
        [
            ("item1", 1, "c.jpg", "cached"),
            ("item1", 0, "b.jpg", "ok"),
            ("item2", 0, "", "forbidden"),
            ("item2", 1, "gone.jpg", "ok"),
            ("item2", 2, "c.jpg", "ok"),
            ("item3", 0, "", "not_found"),
        ],
    )
    index = ImageIndex.from_manifest(manifest, images)
    assert index.path("item1") == images / "b.jpg"
    assert index.path("item2") == images / "c.jpg"
    assert index.path("item3") is None
    assert index.path("unknown") is None
    assert index.coverage(["item1", "item2", "item3"]) == (2, 3)


def test_missing_manifest_gives_empty_index(tmp_path: Path) -> None:
    index = ImageIndex.from_manifest(tmp_path / "none.csv", tmp_path)
    assert len(index) == 0
    assert index.coverage(["item1"]) == (0, 1)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (JPEG, "image/jpeg"),
        (PNG, "image/png"),
        (b"GIF89a" + b"\x00" * 8, "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"<html>", "application/octet-stream"),
    ],
)
def test_media_type_comes_from_magic_bytes(tmp_path: Path, content: bytes, expected: str) -> None:
    path = tmp_path / "image.jpg"
    path.write_bytes(content)
    assert media_type(path) == expected
