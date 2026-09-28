from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

IMAGE_URL_PREFIX = "https://static.ifood-static.com.br/image/upload/t_low/pratos/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) foodsearch-image-prefetch/1.0"
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
MANIFEST_FIELDS = ["itemId", "image_idx", "image_str", "file", "status", "http_status", "bytes"]
IMAGE_SIGNATURES: tuple[bytes, ...] = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a")


@dataclass(frozen=True)
class ImageRef:
    item_id: str
    image_idx: int
    image_str: str


@dataclass(frozen=True)
class FetchResult:
    ref: ImageRef
    file: str
    status: str
    http_status: int | None
    size: int


def image_filename(image_str: str) -> str:
    return hashlib.sha1(image_str.encode("utf-8")).hexdigest() + ".jpg"


def image_url(image_str: str) -> str:
    return IMAGE_URL_PREFIX + image_str


def is_image(body: bytes) -> bool:
    if body.startswith(IMAGE_SIGNATURES):
        return True
    return len(body) >= 12 and body[:4] == b"RIFF" and body[8:12] == b"WEBP"


def extract_image_refs(rows: Iterable[dict]) -> tuple[list[ImageRef], list[str]]:
    seen: set = set()
    refs: list[ImageRef] = []
    items: list[str] = []
    for row in rows:
        item_id = row["itemId"]
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(item_id)
        metadata = json.loads(row["itemMetadata"] or "{}")
        for idx, image_str in enumerate(metadata.get("images") or []):
            if image_str:
                refs.append(ImageRef(item_id, idx, image_str))
    return refs, items


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


def read_items(path: Path) -> Iterator[dict]:
    _raise_csv_field_limit()
    with path.open(newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def download(url: str, timeout: float, retries: int) -> tuple[bytes | None, int | None, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    http_status: int | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read(), response.status, "ok"
        except urllib.error.HTTPError as exc:
            http_status = exc.code
            if exc.code == 404:
                return None, exc.code, "not_found"
            if exc.code == 403:
                return None, exc.code, "forbidden"
            if exc.code not in RETRYABLE_HTTP:
                return None, exc.code, "error"
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            http_status = None
        if attempt < retries:
            time.sleep(min(30.0, 2.0**attempt) + random.uniform(0.0, 0.5))
    return None, http_status, "error"


def fetch_one(ref: ImageRef, out_dir: Path, timeout: float, retries: int) -> FetchResult:
    name = image_filename(ref.image_str)
    target = out_dir / name
    if target.is_file() and target.stat().st_size > 0:
        return FetchResult(ref, name, "cached", None, target.stat().st_size)
    body, http_status, status = download(image_url(ref.image_str), timeout, retries)
    if body is None:
        return FetchResult(ref, "", status, http_status, 0)
    if not is_image(body):
        return FetchResult(ref, "", "not_image", http_status, len(body))
    partial = target.with_name(f"{name}.{os.getpid()}.{ref.image_idx}.part")
    partial.write_bytes(body)
    os.replace(partial, target)
    return FetchResult(ref, name, "ok", http_status, len(body))


def fetch_all(
    refs: Sequence[ImageRef], out_dir: Path, workers: int, timeout: float, retries: int
) -> list[FetchResult]:
    results: list[FetchResult] = []
    step = max(1, len(refs) // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, ref, out_dir, timeout, retries) for ref in refs]
        for done, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if done % step == 0 or done == len(refs):
                print(f"{done}/{len(refs)} images processed", file=sys.stderr, flush=True)
    return results


def write_manifest(path: Path, results: Sequence[FetchResult]) -> None:
    ordered = sorted(results, key=lambda r: (r.ref.item_id, r.ref.image_idx))
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(MANIFEST_FIELDS)
        for r in ordered:
            ref = r.ref
            row = [ref.item_id, ref.image_idx, ref.image_str, r.file, r.status, r.http_status or ""]
            writer.writerow([*row, r.size])
    os.replace(tmp, path)


def summarize(
    results: Sequence[FetchResult], all_refs: Sequence[ImageRef], items: Sequence[str]
) -> str:
    statuses = Counter(r.status for r in results)
    local = {r.ref.item_id for r in results if r.file}
    with_refs = {ref.item_id for ref in all_refs}
    stored = statuses["ok"] + statuses["cached"]
    lines = [
        f"images stored: {stored} / {len(results)}",
        "status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        f"items with a local image: {len(local)} / {len(items)}",
        f"items without images in metadata: {len(items) - len(with_refs)}",
        f"total bytes stored: {sum(r.size for r in results if r.file)}",
    ]
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch every catalog image into a local cache.")
    parser.add_argument("--items", type=Path, default=Path("data/5k_items_curated.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/images"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/image_manifest.csv"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="only fetch the first N images")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    all_refs, items = extract_image_refs(read_items(args.items))
    refs = all_refs if args.limit is None else all_refs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    results = fetch_all(refs, args.out, args.workers, args.timeout, args.retries)
    write_manifest(args.manifest, results)
    print(summarize(results, all_refs, items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
