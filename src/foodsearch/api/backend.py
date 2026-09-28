import json
import logging
import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import openai
import pandas as pd

from foodsearch import paths
from foodsearch.api.schemas import (
    CompareResponse,
    CompareSide,
    FoodPrior,
    ItemCard,
    JudgeSummary,
    LabelPair,
    LabelProgress,
    LabelQueue,
    QueriesResponse,
    QueryMetrics,
    QueryRef,
    Result,
    SearchResponse,
    Source,
    Stage,
    Summary,
    SummaryRow,
    SystemInfo,
)
from foodsearch.cards import card_fields
from foodsearch.data import load_items, load_queries
from foodsearch.eval import human, judge
from foodsearch.eval.freeze import load_freeze
from foodsearch.eval.pooling import ranked
from foodsearch.images import ImageIndex, images_dir
from foodsearch.llm import LLMClient, LLMResponseError, load_cost_log
from foodsearch.pipeline import Pipeline, PipelineResult
from foodsearch.query import UNDERSTANDING_MODEL
from foodsearch.retrievers import Retriever
from foodsearch.runs import Run, load_run
from foodsearch.systems import REGISTRY, load_systems

logger = logging.getLogger("uvicorn.error")

RESULTS_K = 10
ANNOTATOR = "annotator1"
DEMO_SYSTEMS = (
    ("bm25", "BM25", "BM25"),
    ("dense_pointwise", "Pointwise (S9)", "Pointwise"),
    ("main", "Listwise (S7)", "Listwise"),
    ("hybrid", "Hybrid (S8)", "Hybrid"),
)
QUERY_METRICS = ("ndcg5", "ndcg10", "p5", "food_leak5")

T = TypeVar("T")


class DemoError(Exception):
    status = 500


class UnknownSystemError(DemoError):
    status = 404


class NotFoundError(DemoError):
    status = 404


class LiveQueryUnavailableError(DemoError):
    status = 503


class UpstreamError(DemoError):
    status = 502


@dataclass
class _Outcome:
    ranking: list[tuple[str, float]]
    detail: PipelineResult | None = None
    stages: list[dict[str, Any]] = field(default_factory=list)


def _clean(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def load_demo_queries() -> pd.DataFrame:
    if paths.queries_csv().exists():
        return load_queries(paths.queries_csv())
    tags = pd.read_csv(paths.query_types_csv(), dtype=str)
    return pd.DataFrame({"query_id": tags["query_id"], "text": tags["search_term_pt"].str.strip()})


class _FileCache:
    def __init__(self) -> None:
        self._entries: dict[Path, tuple[float, Any]] = {}

    def get(self, path: Path, loader: Callable[[Path], T]) -> T | None:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        cached = self._entries.get(path)
        if cached is None or cached[0] != mtime:
            cached = (mtime, loader(path))
            self._entries[path] = cached
        return cached[1]


class DemoBackend:
    def __init__(
        self,
        items: pd.DataFrame,
        queries: pd.DataFrame,
        systems: dict[str, dict[str, Any]],
        images: ImageIndex,
        query_types: dict[str, str] | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.items = items
        self.rows = items.set_index("item_id", drop=False)
        self.queries = queries
        self.systems = systems
        self.images = images
        self.query_types = query_types or {}
        self._llm = llm
        self._query_ids = dict(zip(queries["text"], queries["query_id"], strict=True))
        self._texts = dict(zip(queries["query_id"], queries["text"], strict=True))
        self._retrievers: dict[str, Retriever] = {}
        self._lock = threading.RLock()
        self._files = _FileCache()

    @classmethod
    def from_artifacts(
        cls, config: Path | None = None, llm: LLMClient | None = None
    ) -> "DemoBackend":
        if not paths.items_parquet().exists():
            raise NotFoundError(
                f"{paths.items_parquet()} not found; run `python scripts/build_artifacts.py` "
                "or copy the artifacts/ folder from a machine that has it"
            )
        query_types: dict[str, str] = {}
        if paths.query_types_csv().exists():
            tags = pd.read_csv(paths.query_types_csv(), dtype=str)
            query_types = dict(zip(tags["query_id"], tags["type"], strict=True))
        return cls(
            items=load_items(paths.items_parquet()),
            queries=load_demo_queries(),
            systems=load_systems(config or paths.configs_dir() / "systems.yaml"),
            images=ImageIndex.default(),
            query_types=query_types,
            llm=llm,
        )

    def image_report(self) -> tuple[bool, str]:
        found, total = self.images.coverage(self.items["item_id"])
        if found:
            return True, f"images: {found} / {total} items have a local image ({images_dir()})"
        return False, (
            f"images: no local image found in {images_dir()}; every card will show the "
            "fallback tile. Copy the artifacts/ folder from a machine that has it, or run "
            "`python scripts/fetch_images.py` on a network that reaches the image CDN"
        )

    def queries_response(self) -> QueriesResponse:
        return QueriesResponse(
            queries=[QueryRef(query_id=q, text=t) for q, t in self._texts.items()],
            systems=[
                SystemInfo(
                    id=sid,
                    label=label,
                    short=short,
                    desc=self.systems[sid].get("description", ""),
                )
                for sid, label, short in DEMO_SYSTEMS
                if sid in self.systems
            ],
            judge_model=self._judge_model(),
        )

    def search(self, query: str, system: str) -> SearchResponse:
        query = query.strip()
        spec = self._spec(system)
        query_id = self._query_ids.get(query)
        outcome = self._outcome(query, query_id, system)
        detail = outcome.detail
        understanding = detail.understanding if detail else None
        return SearchResponse(
            query_id=query_id,
            query=query,
            system=system,
            query_type=self.query_types.get(query_id) if query_id else None,
            intent=detail.intent if detail else None,
            expanded_dishes=list(understanding.dishes_pt) if understanding else [],
            food_prior=(
                FoodPrior(on=detail.prior.on, lambda_=detail.prior.lam, reason=detail.prior.reason)
                if detail and detail.prior
                else None
            ),
            understanding_model=UNDERSTANDING_MODEL if understanding else None,
            stages=[Stage.model_validate(s) for s in outcome.stages],
            results=self._results(outcome, query_id, spec["type"]),
            metrics=self._metrics(query_id),
        )

    def compare(self, query: str, left: str, right: str) -> CompareResponse:
        sides = [self.search(query, system) for system in (left, right)]
        intent = next((s.intent for s in sides if s.intent is not None), None)

        def side(response: SearchResponse) -> CompareSide:
            metrics = response.metrics.get(response.system) if response.metrics else None
            return CompareSide(system=response.system, results=response.results, metrics=metrics)

        return CompareResponse(
            query=sides[0].query,
            query_type=sides[0].query_type,
            intent=intent,
            left=side(sides[0]),
            right=side(sides[1]),
        )

    def _spec(self, system: str) -> dict[str, Any]:
        if system not in self.systems:
            raise UnknownSystemError(f"Unknown system {system!r}; defined: {sorted(self.systems)}")
        return self.systems[system]

    def _retriever(self, system: str) -> Retriever:
        with self._lock:
            if system not in self._retrievers:
                spec = self._spec(system)
                params = dict(spec.get("params") or {})
                if spec["type"] != "bm25" and self._llm is not None:
                    params["llm"] = self._llm
                retriever = REGISTRY[spec["type"]](name=system, **params)
                retriever.fit(self.items)
                self._retrievers[system] = retriever
            return self._retrievers[system]

    def _run(self, system: str) -> Run | None:
        return self._files.get(paths.runs_dir() / f"{system}.json", load_run)

    def _meta(self, system: str) -> dict[str, Any] | None:
        path = paths.run_meta_dir() / f"{system}.json"
        return self._files.get(path, lambda p: json.loads(p.read_text(encoding="utf-8")))

    def _outcome(self, query: str, query_id: str | None, system: str) -> _Outcome:
        run = self._run(system) if query_id else None
        if query_id is None or run is None or query_id not in run:
            return self._live(query, system)
        scores = run[query_id]
        outcome = _Outcome([(iid, scores[iid]) for iid in ranked(scores, RESULTS_K)])
        meta = self._meta(system)
        if meta is not None and query_id in meta.get("queries", {}):
            outcome.stages = meta["queries"][query_id]["stages"]
        retriever = self._retriever(system) if self._spec(system)["type"] == "pipeline" else None
        if isinstance(retriever, Pipeline):
            try:
                outcome.detail = self._detailed(query, retriever)
            except (openai.OpenAIError, LLMResponseError) as e:
                logger.warning("no cached trace for %s on %s: %s", query_id, system, e)
        return outcome

    def _detailed(self, query: str, pipeline: Pipeline) -> PipelineResult:
        with self._lock:
            return pipeline.search_detailed(query, RESULTS_K)

    def _live(self, query: str, system: str) -> _Outcome:
        retriever = self._retriever(system)
        try:
            if isinstance(retriever, Pipeline):
                detail = self._detailed(query, retriever)
                return _Outcome(
                    [(h.item_id, h.score) for h in detail.hits],
                    detail,
                    [asdict(s) for s in detail.stages],
                )
            with self._lock:
                hits = retriever.search(query, RESULTS_K)
            return _Outcome([(h.item_id, h.score) for h in hits])
        except openai.APIError as e:
            raise UpstreamError(f"OpenAI request failed: {e}") from e
        except openai.OpenAIError as e:
            raise LiveQueryUnavailableError(
                "Live query needs OPENAI_API_KEY; only the cached eval queries work without it"
            ) from e
        except LLMResponseError as e:
            raise UpstreamError(str(e)) from e

    def _results(self, outcome: _Outcome, query_id: str | None, kind: str) -> list[Result]:
        judgments = self._judgments().get(query_id, {}) if query_id else {}
        results = []
        for rank, (item_id, score) in enumerate(outcome.ranking, start=1):
            row = self.rows.loc[item_id]
            if outcome.detail is not None:
                sources = [
                    Source(label=label, rank=r)
                    for label, r in outcome.detail.sources.get(item_id, [])
                ]
            elif kind == "pipeline":
                sources = []
            else:
                sources = [Source(label=kind, rank=rank)]
            grade = judgments.get(item_id)
            results.append(
                Result(
                    **card_fields(row),
                    rank=rank,
                    score=score,
                    sources=sources,
                    judge_grade=grade.grade if grade else None,
                    judge_reason=grade.reason if grade else None,
                    doc_text=row["doc"],
                )
            )
        return results

    def _freeze(self) -> dict[str, Any] | None:
        return self._files.get(paths.eval_freeze_json(), load_freeze)

    def _judge_model(self) -> str:
        freeze = self._freeze()
        return freeze["judge_model"] if freeze else judge.JUDGE_MODEL

    def _judge_result(self) -> judge.JudgeResult | None:
        freeze = self._freeze()
        if freeze is None:
            return None
        path = judge.judgments_path(
            paths.judgments_dir(), freeze["judge_model"], freeze["prompt_lang"], "all"
        )
        return self._files.get(path, judge.load_judgments)

    def _judgments(self) -> dict[str, dict[str, judge.Judgment]]:
        result = self._judge_result()
        return result.grades if result else {}

    def _per_query_metrics(self) -> dict[str, dict[str, QueryMetrics]] | None:
        def load(path: Path) -> dict[str, dict[str, QueryMetrics]]:
            table = pd.read_csv(path, dtype={"query_id": str, "system": str})
            out: dict[str, dict[str, QueryMetrics]] = {}
            for row in table.to_dict("records"):
                out.setdefault(row["query_id"], {})[row["system"]] = QueryMetrics(
                    **{m: _clean(row.get(m)) for m in QUERY_METRICS}
                )
            return out

        return self._files.get(paths.reports_dir() / "per_query.csv", load)

    def _metrics(self, query_id: str | None) -> dict[str, QueryMetrics] | None:
        table = self._per_query_metrics()
        if query_id is None or table is None:
            return None
        return table.get(query_id)

    def _queue(self) -> list[human.LabelPair] | None:
        return self._files.get(paths.labels_dir() / "queue.csv", human.load_queue)

    def _labels_path(self) -> Path:
        return paths.labels_dir() / "human.csv"

    def _labelled(self) -> set[str]:
        labels = human.load_labels(self._labels_path())
        mine = labels[labels["annotator"] == ANNOTATOR]
        return {
            f"{q}:{i}:{int(p)}"
            for q, i, p in zip(mine["query_id"], mine["item_id"], mine["pass"], strict=True)
        }

    def _require_queue(self) -> list[human.LabelPair]:
        queue = self._queue()
        if queue is None:
            raise NotFoundError("No label queue yet; run `foodsearch labels queue`")
        return queue

    def label_progress(self) -> LabelProgress:
        queue = self._queue() or []
        labelled = self._labelled()
        return LabelProgress(done=sum(p.pair_id in labelled for p in queue), total=len(queue))

    def label_queue(self) -> LabelQueue:
        queue = self._require_queue()
        labelled = self._labelled()
        pairs = [
            LabelPair(
                pair_id=p.pair_id,
                query=self._texts[p.query_id],
                item=ItemCard(**card_fields(self.rows.loc[p.item_id])),
            )
            for p in queue
            if p.pair_id not in labelled
        ]
        return LabelQueue(pairs=pairs, done=len(queue) - len(pairs), total=len(queue))

    def record_label(self, pair_id: str, grade: int) -> LabelProgress:
        queue = self._require_queue()
        if pair_id not in {p.pair_id for p in queue}:
            raise NotFoundError(f"pair_id {pair_id!r} is not in the label queue")
        with self._lock:
            human.record_label(self._labels_path(), pair_id, grade, ANNOTATOR)
        return self.label_progress()

    def labels_file(self) -> Path | None:
        path = self._labels_path()
        return path if path.exists() else None

    def summary(self) -> Summary:
        result = self._judge_result()
        freeze = self._freeze()
        kappa = freeze["agreement"]["values"].get("kappa_quadratic") if freeze else None
        cost = None
        if paths.cost_log().exists():
            cost = float(load_cost_log(paths.cost_log())["usd"].sum())
        return Summary(
            n_queries=len(self.queries),
            rows=self._summary_rows(),
            judge=JudgeSummary(
                model=self._judge_model(),
                kappa=_clean(kappa),
                judged=result.n_judged if result else 0,
                unjudged=len(result.failures) if result else 0,
            ),
            cost_usd=cost,
            label=self.label_progress(),
        )

    def _summary_rows(self) -> list[SummaryRow]:
        table = self._files.get(paths.reports_dir() / "metrics.csv", pd.read_csv)
        if table is None:
            return []
        rows = []
        for system, sub in table.groupby("system", sort=False):
            by_metric = sub.set_index("metric")
            values: dict[str, list[float | None] | None] = {}
            for metric in QUERY_METRICS:
                if metric not in by_metric.index or _clean(by_metric.at[metric, "mean"]) is None:
                    values[metric] = None
                else:
                    values[metric] = [_clean(by_metric.at[metric, c]) for c in ("mean", "lo", "hi")]
            rows.append(SummaryRow(system=str(system), **values))
        return rows
