from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from foodsearch import paths

STORED_STATUSES = frozenset({"ok", "cached"})
SIGNATURES = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
FALLBACK_MEDIA_TYPE = "application/octet-stream"


def images_dir() -> Path:
    return paths.artifacts_dir() / "images"


def manifest_csv() -> Path:
    return paths.artifacts_dir() / "image_manifest.csv"


def media_type(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(12)
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    for signature, kind in SIGNATURES:
        if head.startswith(signature):
            return kind
    return FALLBACK_MEDIA_TYPE


class ImageIndex:
    def __init__(self, candidates: Mapping[str, Sequence[Path]]) -> None:
        self._candidates = {item_id: list(files) for item_id, files in candidates.items()}

    @classmethod
    def from_manifest(cls, manifest: Path, directory: Path) -> "ImageIndex":
        if not manifest.exists():
            return cls({})
        rows = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        rows = rows[rows["status"].isin(STORED_STATUSES) & (rows["file"] != "")]
        rows = rows.assign(order=rows["image_idx"].astype(int)).sort_values(["itemId", "order"])
        candidates: dict[str, list[Path]] = {}
        for item_id, file in zip(rows["itemId"], rows["file"], strict=True):
            candidates.setdefault(item_id, []).append(directory / file)
        return cls(candidates)

    @classmethod
    def default(cls) -> "ImageIndex":
        return cls.from_manifest(manifest_csv(), images_dir())

    def __len__(self) -> int:
        return len(self._candidates)

    def path(self, item_id: str) -> Path | None:
        return next((p for p in self._candidates.get(item_id, ()) if p.is_file()), None)

    def coverage(self, item_ids: Iterable[str]) -> tuple[int, int]:
        ids = list(item_ids)
        return sum(self.path(item_id) is not None for item_id in ids), len(ids)
