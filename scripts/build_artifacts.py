import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import fetch_images
import openai

from foodsearch import cli, paths
from foodsearch.data import load_items
from foodsearch.eval.freeze import load_freeze
from foodsearch.images import ImageIndex, images_dir, manifest_csv
from foodsearch.systems import load_systems


@dataclass
class Report:
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build every artifact the evaluation and the demo need from data/."
    )
    parser.add_argument("--root", type=Path, help="project root (default: FOODSEARCH_ROOT or cwd)")
    parser.add_argument("--config", type=Path, help="systems file (default: configs/systems.yaml)")
    parser.add_argument("--systems", nargs="+", help="systems to run (default: all configured)")
    parser.add_argument("--force", action="store_true", help="rebuild outputs that already exist")
    parser.add_argument("--skip-images", action="store_true", help="don't download images")
    parser.add_argument("--workers", type=int, default=16, help="image download threads")
    return parser.parse_args(argv)


def _step(report: Report, name: str, action: Callable[[], None]) -> bool:
    print(f"\n== {name}", flush=True)
    try:
        action()
    except SystemExit as e:
        if e.code not in (None, 0):
            report.failed.append(f"{name}: {e.code}")
            return False
    report.done.append(name)
    return True


def _skip(report: Report, name: str, reason: str) -> None:
    print(f"\n== {name}: skipped, {reason}", flush=True)
    report.skipped.append(f"{name} ({reason})")


def _run_system(report: Report, system: str, config: Path) -> None:
    name = f"run {system}"
    try:
        _step(report, name, lambda: cli.main(["run", "--system", system, "--config", str(config)]))
    except openai.APIError as e:
        report.failed.append(f"{name}: OpenAI request failed ({e})")
    except openai.OpenAIError:
        _skip(report, name, "needs OPENAI_API_KEY for calls missing from artifacts/cache")


def build(args: argparse.Namespace) -> Report:
    if args.root is not None:
        os.environ[paths.ROOT_ENV_VAR] = str(args.root.resolve())
    config = args.config or paths.configs_dir() / "systems.yaml"
    report = Report()

    missing = [p for p in (paths.items_csv(), paths.queries_csv(), config) if not p.exists()]
    if missing:
        sys.exit("missing inputs: " + ", ".join(str(p) for p in missing))
    systems = load_systems(config)
    selected = args.systems or list(systems)
    unknown = sorted(set(selected) - set(systems))
    if unknown:
        sys.exit(f"unknown systems {unknown}; defined in {config}: {sorted(systems)}")

    if paths.items_parquet().exists() and not args.force:
        _skip(report, "index", f"{paths.items_parquet()} exists")
    else:
        _step(report, "index", lambda: cli.main(["index"]))

    if args.skip_images:
        _skip(report, "images", "--skip-images")
    elif manifest_csv().exists() and not args.force:
        _skip(report, "images", f"{manifest_csv()} exists")
    else:
        image_argv = [
            f"--items={paths.items_csv()}",
            f"--out={images_dir()}",
            f"--manifest={manifest_csv()}",
            f"--workers={args.workers}",
        ]
        _step(report, "images", lambda: sys.exit(fetch_images.main(image_argv)))

    for system in selected:
        if (paths.runs_dir() / f"{system}.json").exists() and not args.force:
            _skip(report, f"run {system}", "run exists")
        else:
            _run_system(report, system, config)

    runs = [s for s in systems if (paths.runs_dir() / f"{s}.json").exists()]
    if load_freeze(paths.eval_freeze_json()) is not None:
        _skip(report, "pool", "the evaluation is frozen")
    elif paths.pool_json().exists() and not args.force:
        _skip(report, "pool", f"{paths.pool_json()} exists")
    elif not runs:
        _skip(report, "pool", "no runs")
    else:
        _step(report, "pool", lambda: cli.main(["pool", "--systems", *runs]))

    if paths.qrels_json().exists() and runs:
        _step(report, "eval", lambda: cli.main(["eval", "--systems", *runs]))
    else:
        _skip(report, "eval", f"{paths.qrels_json()} not built yet")
    return report


def status_lines(report: Report) -> list[str]:
    lines = ["", "== status"]
    lines += [f"  done     {name}" for name in report.done]
    lines += [f"  skipped  {name}" for name in report.skipped]
    lines += [f"  FAILED   {name}" for name in report.failed]
    if paths.items_parquet().exists():
        items = load_items(paths.items_parquet())
        found, total = ImageIndex.default().coverage(items["item_id"])
        lines.append(f"  images   {found} / {total} items have a local image in {images_dir()}")
    lines += [
        "",
        "Manual steps left (human labels and the paid judge stay manual on purpose):",
        "  foodsearch labels queue, then label in the demo (Label view) or via labels sheet/import",
        "  foodsearch judge --subset human --lang en, then foodsearch agreement --lang en",
        "  foodsearch freeze --lang en, foodsearch judge --subset all, then rerun this script",
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    report = build(parse_args(argv))
    print("\n".join(status_lines(report)))
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
