import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from foodsearch import paths
from foodsearch.data import build_items, load_items, load_queries, save_items
from foodsearch.retrievers import Retriever
from foodsearch.runs import hits_to_run, save_run
from foodsearch.systems import build_system


def _load_items() -> pd.DataFrame:
    path = paths.items_parquet()
    if not path.exists():
        sys.exit(f"{path} not found; run `foodsearch index` first")
    return load_items(path)


def _fitted_system(name: str, config: Path, items: pd.DataFrame) -> Retriever:
    system = build_system(name, config)
    system.fit(items)
    return system


def cmd_index(args: argparse.Namespace) -> None:
    items = build_items(paths.items_csv())
    save_items(items, paths.items_parquet())
    print(f"{len(items)} items written to {paths.items_parquet()}")


def cmd_search(args: argparse.Namespace) -> None:
    items = _load_items()
    system = _fitted_system(args.system, args.config, items)
    names = items.set_index("item_id")[["name", "l0"]]
    hits = system.search(args.query, k=args.k)
    if not hits:
        print("no results")
    for rank, hit in enumerate(hits, start=1):
        name, l0 = names.loc[hit.item_id]
        print(f"{rank:>3}  {hit.score:8.4f}  {hit.item_id}  {name}  [{l0}]")


def cmd_run(args: argparse.Namespace) -> None:
    items = _load_items()
    system = _fitted_system(args.system, args.config, items)
    queries = load_queries(paths.queries_csv())
    hits = {q.query_id: system.search(q.text, k=args.k) for q in queries.itertuples(index=False)}
    out = paths.runs_dir() / f"{args.system}.json"
    save_run(hits_to_run(hits), out)
    empty = sum(not h for h in hits.values())
    print(f"{len(hits)} queries ({empty} with no hits) written to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foodsearch")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="parse the item CSV into artifacts/items.parquet")
    index.set_defaults(func=cmd_index)

    for command, func, k, help_text in [
        ("search", cmd_search, 10, "search one query"),
        ("run", cmd_run, 100, "run a system on all queries into artifacts/runs/<system>.json"),
    ]:
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--system", required=True)
        p.add_argument("-k", type=int, default=k)
        p.add_argument("--config", type=Path, default=paths.configs_dir() / "systems.yaml")
        p.set_defaults(func=func)
        if command == "search":
            p.add_argument("query")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
