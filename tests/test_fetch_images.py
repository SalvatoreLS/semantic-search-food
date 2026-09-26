import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_images.py"
spec = importlib.util.spec_from_file_location("fetch_images", SCRIPT)
fetch_images = importlib.util.module_from_spec(spec)
sys.modules["fetch_images"] = fetch_images
spec.loader.exec_module(fetch_images)


def _row(item_id: str, images: list) -> dict:
    return {"itemId": item_id, "itemMetadata": json.dumps({"name": "x", "images": images})}


def test_extract_image_refs_dedupes_items_and_keeps_every_image() -> None:
    rows = [
        _row("a", ["m/1.jpg", "m/2.jpg"]),
        _row("a", ["m/1.jpg", "m/2.jpg"]),
        _row("b", []),
        _row("c", ["m/3.jpg"]),
    ]
    refs, items = fetch_images.extract_image_refs(rows)
    assert items == ["a", "b", "c"]
    assert [(r.item_id, r.image_idx, r.image_str) for r in refs] == [
        ("a", 0, "m/1.jpg"),
        ("a", 1, "m/2.jpg"),
        ("c", 0, "m/3.jpg"),
    ]


def test_image_filename_is_stable_sha1() -> None:
    name = fetch_images.image_filename("m/1.jpg")
    assert name == fetch_images.image_filename("m/1.jpg")
    assert name.endswith(".jpg") and len(name) == 44
    assert name != fetch_images.image_filename("m/2.jpg")


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b"\xff\xd8\xff\xe0rest", True),
        (b"\x89PNG\r\n\x1a\nrest", True),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", True),
        (b"<html>denied</html>", False),
        (b"", False),
    ],
)
def test_is_image(body: bytes, expected: bool) -> None:
    assert fetch_images.is_image(body) is expected


def test_fetch_one_skips_cached_file(tmp_path: Path) -> None:
    ref = fetch_images.ImageRef("a", 0, "m/1.jpg")
    (tmp_path / fetch_images.image_filename("m/1.jpg")).write_bytes(b"\xff\xd8\xffdata")
    result = fetch_images.fetch_one(ref, tmp_path, timeout=1.0, retries=0)
    assert result.status == "cached"
    assert result.size == 7
