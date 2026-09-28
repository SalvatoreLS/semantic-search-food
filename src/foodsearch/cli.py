import argparse
import json
import os
import socket
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn

from foodsearch import paths
from foodsearch.api.app import create_app
from foodsearch.api.backend import DemoError
from foodsearch.data import build_items, load_items, load_queries, save_items
from foodsearch.device import resolve_device
from foodsearch.eval import human, judge, metrics
from foodsearch.eval.agreement import agreement, intra_agreement
from foodsearch.eval.freeze import check_prompt_unchanged, load_freeze, write_freeze
from foodsearch.eval.pooling import build_pool, load_pool, save_pool
from foodsearch.eval.prompts import LANGS
from foodsearch.llm import LLMClient, default_client
from foodsearch.pipeline import Pipeline, PipelineResult
from foodsearch.retrievers import Hit, Retriever
from foodsearch.runs import Run, hits_to_run, load_run, save_run
from foodsearch.systems import build_system

SERVE_HOST = "127.0.0.1"


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


def _print_trace(result: PipelineResult) -> None:
    if result.understanding is not None:
        print(f"intent: {result.intent}; dishes: {', '.join(result.understanding.dishes_pt)}")
    elif result.intent is not None:
        print(f"intent: {result.intent}")
    for stage in result.stages:
        model = f" ({stage.model})" if stage.model else ""
        cost = f"{stage.ms:.0f} ms, ${stage.cost_usd:.4f}"
        print(f"  {stage.name}{model}: {stage.detail or ''} [{cost}]")


def cmd_search(args: argparse.Namespace) -> None:
    items = _load_items()
    system = _fitted_system(args.system, args.config, items)
    names = items.set_index("item_id")[["name", "l0"]]
    if isinstance(system, Pipeline):
        result = system.search_detailed(args.query, k=args.k)
        _print_trace(result)
        hits = result.hits
    else:
        hits = system.search(args.query, k=args.k)
    if not hits:
        print("no results")
    for rank, hit in enumerate(hits, start=1):
        name, l0 = names.loc[hit.item_id]
        print(f"{rank:>3}  {hit.score:8.4f}  {hit.item_id}  {name}  [{l0}]")


def _timed_search(system: Retriever, query: str, k: int) -> tuple[list[Hit], dict[str, Any]]:
    start = time.perf_counter()
    if isinstance(system, Pipeline):
        result = system.search_detailed(query, k=k)
        hits, stages = result.hits, [asdict(s) for s in result.stages]
    else:
        hits, stages = system.search(query, k=k), []
    ms = (time.perf_counter() - start) * 1000
    return hits, {"ms": round(ms, 1), "stages": stages}


def cmd_run(args: argparse.Namespace) -> None:
    items = _load_items()
    system = _fitted_system(args.system, args.config, items)
    queries = load_queries(paths.queries_csv())
    costs_before = len(default_client().costs.calls)
    hits: dict[str, list[Hit]] = {}
    per_query: dict[str, dict[str, Any]] = {}
    for q in queries.itertuples(index=False):
        hits[q.query_id], per_query[q.query_id] = _timed_search(system, q.text, args.k)
    out = paths.runs_dir() / f"{args.system}.json"
    save_run(hits_to_run(hits), out)
    new_calls = default_client().costs.calls[costs_before:]
    meta = {
        "system": args.system,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": resolve_device(),
        "paid_calls": len(new_calls),
        "usd": sum(c.usd for c in new_calls),
        "queries": per_query,
    }
    meta_path = paths.run_meta_dir() / f"{args.system}.json"
    kept = _keep_cold_meta(meta_path, meta)
    empty = sum(not h for h in hits.values())
    print(
        f"{len(hits)} queries ({empty} with no hits) written to {out}; "
        f"{meta['paid_calls']} paid calls, ${meta['usd']:.4f}"
    )
    if kept:
        print(f"cached rerun: kept the cold-run timing in {meta_path}")


def _keep_cold_meta(path: Path, meta: dict[str, Any]) -> bool:
    if path.exists() and meta["paid_calls"] == 0:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("paid_calls", 0) > 0:
            return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return False


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        sys.exit(f"{path} not found; {hint}")
    return path


def _load_runs(systems: Sequence[str] | None) -> dict[str, Run]:
    files = sorted(paths.runs_dir().glob("*.json"))
    runs = {f.stem: load_run(f) for f in files if not systems or f.stem in systems}
    missing = set(systems or []) - set(runs)
    if missing or not runs:
        sys.exit(f"missing runs in {paths.runs_dir()}: {sorted(missing) or 'none found'}")
    return runs


def _label_queue() -> list[human.LabelPair]:
    queue_path = _require(paths.labels_dir() / "queue.csv", "run `foodsearch labels queue`")
    return human.load_queue(queue_path)


def _human_judgments_path(model: str, lang: str) -> Path:
    return judge.judgments_path(paths.judgments_dir(), model, lang, "human")


def _agreement_inputs(model: str, lang: str) -> tuple[judge.Qrels, judge.Qrels]:
    labels = human.load_labels(paths.labels_dir() / "human.csv")
    if labels.empty:
        sys.exit("no human labels yet in artifacts/labels/human.csv")
    hint = f"run `foodsearch judge --subset human --model {model} --lang {lang}`"
    judged_path = _require(_human_judgments_path(model, lang), hint)
    return human.human_qrels(labels), judge.grades_of(judge.load_judgments(judged_path))


def cmd_pool(args: argparse.Namespace) -> None:
    pool = build_pool(_load_runs(args.systems), depth=args.depth)
    save_pool(pool, paths.pool_json())
    sizes = [len(items) for items in pool.best_rank.values()]
    print(
        f"pool of {sum(sizes)} pairs over {len(sizes)} queries "
        f"(mean {sum(sizes) / max(len(sizes), 1):.1f}/query) from {pool.systems} "
        f"written to {paths.pool_json()}"
    )


def cmd_labels(args: argparse.Namespace) -> None:
    labels_dir = paths.labels_dir()
    if args.action == "queue":
        pool = load_pool(_require(paths.pool_json(), "run `foodsearch pool` first"))
        query_ids = human.load_label_query_ids(paths.human_label_queries_csv())
        queue = human.build_label_queue(human.sample_pairs(pool, query_ids))
        queue_path = labels_dir / "queue.csv"
        try:
            human.save_queue(queue, queue_path)
        except FileExistsError as e:
            sys.exit(str(e))
        repeats = sum(p.pass_ == 2 for p in queue)
        print(f"{len(queue) - repeats} pairs + {repeats} repeats written to {queue_path}")
    elif args.action == "sheet":
        queries = load_queries(paths.queries_csv())
        human.export_sheet(_label_queue(), queries, _load_items(), labels_dir / "sheet.csv")
        print(f"label sheet written to {labels_dir / 'sheet.csv'}")
    else:
        sheet = _require(args.sheet, "pass the filled sheet with --sheet")
        n = human.import_sheet(sheet, labels_dir / "human.csv", args.annotator)
        print(f"{n} graded rows imported into {labels_dir / 'human.csv'}")


def cmd_judge(args: argparse.Namespace) -> None:
    freeze = load_freeze(paths.eval_freeze_json())
    if args.subset == "all":
        if freeze is None:
            sys.exit("judging the full pool needs artifacts/eval_freeze.json (`foodsearch freeze`)")
        check_prompt_unchanged(freeze)
        model, lang = freeze["judge_model"], freeze["prompt_lang"]
        if args.model not in (None, model) or args.lang not in (None, lang):
            sys.exit(f"the freeze fixes the judge to {model} ({lang}); drop --model/--lang")
        pool = load_pool(_require(paths.pool_json(), "run `foodsearch pool` first"))
        pairs = pool.pairs()
    else:
        model, lang = args.model or judge.JUDGE_MODEL, args.lang or "en"
        pairs = [(p.query_id, p.item_id) for p in _label_queue() if p.pass_ == 1]
    queries = load_queries(paths.queries_csv())
    judge_pairs = judge.make_pairs(pairs, queries, _load_items())
    client = LLMClient(paths.cache_dir(), paths.cost_log())
    result = judge.judge_pairs(client, judge_pairs, model, lang, workers=args.workers)
    out = judge.judgments_path(paths.judgments_dir(), model, lang, args.subset)
    judge.save_judgments(result, out, args.subset)
    failed = len(result.failures)
    print(f"{result.n_judged} pairs judged by {model} ({lang}), {failed} failed -> {out}")
    for failure in result.failures:
        print(f"  unjudged {failure['query_id']} {failure['item_id']}: {failure['error']}")
    if args.subset == "all":
        judge.write_qrels(result, paths.qrels_json())
        print(f"qrels written to {paths.qrels_json()}")


def cmd_agreement(args: argparse.Namespace) -> None:
    human_grades, judge_grades = _agreement_inputs(args.model, args.lang)
    report = agreement(human_grades, judge_grades)
    labels = human.load_labels(paths.labels_dir() / "human.csv")
    payload = {
        "judge_model": args.model,
        "prompt_lang": args.lang,
        **report.to_dict(),
        "intra_annotator": intra_agreement(human.intra_pairs(labels)),
    }
    out = paths.reports_dir() / f"agreement_{args.model}_{args.lang}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"{report.n_pairs} pairs over {report.n_queries} queries, {args.model} ({args.lang})")
    for name, value in report.values.items():
        lo, hi = report.ci[name]
        print(f"  {name:<18} {value:6.3f}  [{lo:6.3f}, {hi:6.3f}]")
    print("  confusion (rows human 0-3, cols judge 0-3):")
    for row in report.confusion:
        print("   " + " ".join(f"{c:4d}" for c in row))
    verdict = "passed" if report.gate_passed else "FAILED"
    print(f"  gate (kappa_quadratic and kappa_binary >= 0.6): {verdict}")
    print(f"written to {out}")


def cmd_freeze(args: argparse.Namespace) -> None:
    human_grades, judge_grades = _agreement_inputs(args.model, args.lang)
    pool = load_pool(_require(paths.pool_json(), "run `foodsearch pool` first"))
    try:
        write_freeze(
            paths.eval_freeze_json(),
            judge_model=args.model,
            lang=args.lang,
            report=agreement(human_grades, judge_grades),
            queue=_label_queue(),
            pool=pool,
        )
    except (FileExistsError, ValueError) as e:
        sys.exit(str(e))
    print(f"evaluation frozen in {paths.eval_freeze_json()}")


def cmd_eval(args: argparse.Namespace) -> None:
    runs = _load_runs(args.systems)
    if args.qrels == "human":
        qrels = human.human_qrels(human.load_labels(paths.labels_dir() / "human.csv"))
        suffix = "_human"
    else:
        qrels = judge.load_qrels(
            _require(paths.qrels_json(), "run `foodsearch judge --subset all`")
        )
        suffix = ""
    if not qrels:
        sys.exit("qrels are empty")
    items = _load_items()
    tags = pd.read_csv(paths.query_types_csv(), dtype={"query_id": str})
    table = metrics.per_query(
        runs,
        qrels,
        dict(zip(items["item_id"], items["is_food"], strict=True)),
        dict(zip(tags["query_id"], tags["type"], strict=True)),
    )
    pairs = metrics.comparison_pairs(list(runs), args.against)
    out = paths.reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    summary = metrics.summarize(table)
    outputs = {
        f"per_query{suffix}.csv": table,
        f"metrics{suffix}.csv": summary,
        f"per_type{suffix}.csv": metrics.per_type(table, tags),
        f"comparisons{suffix}.csv": metrics.compare(table, pairs) if pairs else pd.DataFrame(),
    }
    for name, frame in outputs.items():
        frame.to_csv(out / name, index=False)
    gaps = metrics.gap_counts(qrels)
    wide = summary.pivot(index="system", columns="metric", values="mean")[metrics.METRICS]
    print(wide.round(3).to_string())
    print(
        f"{gaps['queries']} queries: {gaps['no_graded_item']} with no item graded > 0 "
        f"(excluded from nDCG), {gaps['no_relevant_item']} with no item graded >= 2 "
        f"(excluded from MRR)"
    )
    print(f"reports written to {out}: {', '.join(outputs)}")


def port_is_free(port: int, host: str = SERVE_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def next_free_port(port: int, attempts: int = 50) -> int | None:
    candidates = range(port + 1, min(port + 1 + attempts, 65536))
    return next((p for p in candidates if port_is_free(p)), None)


def confirm(question: str) -> bool:
    try:
        answer = input(f"{question} [Y/n] ").strip().lower()
    except EOFError:
        return False
    return answer in ("", "y", "yes")


def resolve_serve_port(port: int) -> int:
    if port_is_free(port):
        return port
    alternative = next_free_port(port)
    if alternative is None:
        sys.exit(f"port {port} is in use and no free port was found nearby; pass --port")
    question = f"port {port} is in use. Serve on {alternative} instead?"
    if not sys.stdin.isatty() or not confirm(question):
        sys.exit(f"port {port} is in use; pass --port")
    return alternative


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        app = create_app(config=args.config)
    except DemoError as e:
        sys.exit(str(e))
    port = resolve_serve_port(args.port)
    print(f"FoodSearch demo on http://{SERVE_HOST}:{port}")
    uvicorn.run(app, host=SERVE_HOST, port=port)


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

    pool = sub.add_parser("pool", help="pool the top-k of every run into artifacts/pool.json")
    pool.add_argument("--systems", nargs="+")
    pool.add_argument("--depth", type=int, default=10)
    pool.set_defaults(func=cmd_pool)

    labels = sub.add_parser("labels", help="human-label queue, CSV sheet export and import")
    labels.add_argument("action", choices=["queue", "sheet", "import"])
    labels.add_argument("--sheet", type=Path, default=paths.labels_dir() / "sheet.csv")
    labels.add_argument("--annotator", default="annotator1")
    labels.set_defaults(func=cmd_labels)

    judge_p = sub.add_parser("judge", help="grade pooled pairs with the LLM judge")
    judge_p.add_argument("--subset", choices=["human", "all"], required=True)
    judge_p.add_argument("--model", help=f"human subset only (default {judge.JUDGE_MODEL})")
    judge_p.add_argument("--lang", choices=LANGS, help="human subset only (default en)")
    judge_p.add_argument("--workers", type=int, default=8)
    judge_p.set_defaults(func=cmd_judge)

    for command, func, help_text in [
        ("agreement", cmd_agreement, "judge vs human agreement on the labelled subset"),
        ("freeze", cmd_freeze, "freeze rubric, prompt and judge into artifacts/eval_freeze.json"),
    ]:
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--model", default=judge.JUDGE_MODEL)
        p.add_argument("--lang", choices=LANGS, required=True)
        p.set_defaults(func=func)

    ev = sub.add_parser("eval", help="metrics, CIs and paired tests into reports/")
    ev.add_argument("--systems", nargs="+")
    ev.add_argument("--qrels", choices=["judge", "human"], default="judge")
    ev.add_argument("--against", nargs="+", default=["bm25", "dense_pointwise"])
    ev.set_defaults(func=cmd_eval)

    serve = sub.add_parser("serve", help=f"demo UI and API on http://{SERVE_HOST}")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--config", type=Path, default=paths.configs_dir() / "systems.yaml")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
