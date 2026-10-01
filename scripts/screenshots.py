import argparse
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlencode

from foodsearch import paths
from foodsearch.systems import COMPARE_SYSTEM, HEADLINE_SYSTEM

DEFAULT_QUERY = "Comida para piquenique no parque"
CHROME_NAMES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)
SERVER_TIMEOUT_S = 120.0
SERVE_CODE = "from foodsearch.cli import main; main()"
VIEW_HEIGHTS = {"search": 1480, "compare": 1280, "about": 1560}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screenshot the demo views with headless Chrome and foodsearch serve."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY, help="eval query to show")
    parser.add_argument("--out", type=Path, help="output folder (default: artifacts/screenshots)")
    parser.add_argument("--chrome", help="Chrome or Chromium binary (default: first found on PATH)")
    parser.add_argument("--width", type=int, default=1440, help="window width in px")
    parser.add_argument("--port", type=int, help="server port (default: a free one)")
    parser.add_argument("--budget-ms", type=int, default=15000, help="virtual time budget per page")
    return parser.parse_args(argv)


def view_urls(base: str, query: str) -> dict[str, str]:
    compare = {"q": query, "left": COMPARE_SYSTEM, "right": HEADLINE_SYSTEM}
    return {
        "search": f"{base}/#search?{urlencode({'q': query, 'system': HEADLINE_SYSTEM})}",
        "compare": f"{base}/#compare?{urlencode(compare)}",
        "about": f"{base}/#about",
    }


def find_chrome(explicit: str | None = None) -> str:
    candidates = (explicit,) if explicit else CHROME_NAMES
    for name in candidates:
        found = shutil.which(name) if name else None
        if found:
            return found
    tried = ", ".join(c for c in candidates if c)
    raise SystemExit(f"no Chrome binary found (tried {tried}); pass --chrome")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(
    base: str, server: subprocess.Popen[bytes], timeout_s: float = SERVER_TIMEOUT_S
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise SystemExit(f"foodsearch serve exited ({server.returncode}), see serve.log")
        try:
            with urllib.request.urlopen(f"{base}/api/queries", timeout=2):
                return
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    raise SystemExit(f"foodsearch serve not ready after {timeout_s:.0f}s")


def shoot(
    chrome: str, url: str, out: Path, size: tuple[int, int], budget_ms: int, profile: Path
) -> None:
    done = subprocess.run(
        [
            chrome,
            "--headless=new",
            f"--user-data-dir={profile}",
            "--hide-scrollbars",
            "--no-first-run",
            "--force-prefers-reduced-motion",
            f"--window-size={size[0]},{size[1]}",
            f"--virtual-time-budget={budget_ms}",
            f"--screenshot={out}",
            url,
        ],
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or not out.is_file():
        raise SystemExit(f"Chrome failed on {url}:\n{done.stderr.strip()}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    chrome = find_chrome(args.chrome)
    out_dir = args.out or paths.artifacts_dir() / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or free_port()
    base = f"http://127.0.0.1:{port}"
    with (out_dir / "serve.log").open("wb") as log:
        server = subprocess.Popen(
            [sys.executable, "-c", SERVE_CODE, "serve", "--port", str(port)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(base, server)
            with tempfile.TemporaryDirectory(prefix="foodsearch-chrome-") as profile:
                for view, url in view_urls(base, args.query).items():
                    target = out_dir / f"demo_{view}.png"
                    size = (args.width, VIEW_HEIGHTS[view])
                    shoot(chrome, url, target, size, args.budget_ms, Path(profile))
                    print(f"{view}: {target}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
