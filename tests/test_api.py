import io
import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import openai
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from foodsearch import paths
from foodsearch.api.app import create_app
from foodsearch.api.backend import DemoBackend, load_demo_queries
from foodsearch.eval.human import LabelPair, save_queue
from foodsearch.images import ImageIndex
from foodsearch.llm import LLMClient
from foodsearch.runs import save_run

UNDERSTANDING = {
    "intent": "occasion",
    "dishes_pt": ["hambúrguer", "batata frita"],
    "keywords_pt": [],
    "meal_shift": "snack",
    "dietary": [],
}
QUERY = "Comida para piquenique"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
SYSTEMS: dict[str, dict[str, Any]] = {
    "bm25": {"type": "bm25", "description": "BM25 baseline"},
    "main": {
        "type": "pipeline",
        "description": "Main pipeline",
        "params": {
            "dense": {"backend": "openai", "model": "text-embedding-3-small"},
            "understanding": "llm",
            "expand": True,
            "prior_lambda": 0.5,
            "rerank": {"type": "listwise", "depth": 3, "top": 2},
        },
    },
}
MAIN_STAGES = [
    {"name": "query understanding", "model": "gpt-4.1-mini", "detail": "intent occasion"},
    {"name": "listwise rerank", "model": "gpt-4.1-mini", "detail": "top 3"},
]


def _handler(messages: list[dict[str, str]]) -> str:
    if "ranking" in messages[0]["content"]:
        return json.dumps({"ranking": [2, 1]})
    return json.dumps(UNDERSTANDING)


def _no_key(messages: list[dict[str, str]]) -> str:
    raise openai.OpenAIError("The api_key client option must be set")


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(paths.ROOT_ENV_VAR, str(tmp_path))
    save_run({"q001": {"pizza": 3.0, "burger": 2.0, "fries": 1.0}}, paths.runs_dir() / "bm25.json")
    save_run({"q001": {"burger": 0.9, "fries": 0.8}}, paths.runs_dir() / "main.json")
    meta = {
        "system": "main",
        "queries": {
            "q001": {
                "ms": 900.0,
                "stages": [
                    {**s, "ms": 450.0, "cost_usd": 0.001, "llm_calls": 1, "failures": 0}
                    for s in MAIN_STAGES
                ],
            }
        },
    }
    paths.run_meta_dir().mkdir(parents=True)
    (paths.run_meta_dir() / "main.json").write_text(json.dumps(meta), encoding="utf-8")
    images = paths.artifacts_dir() / "images"
    images.mkdir()
    (images / "burger.jpg").write_bytes(PNG)
    pd.DataFrame(
        [("burger", 0, "burger.jpg", "ok"), ("pizza", 0, "", "forbidden")],
        columns=["itemId", "image_idx", "file", "status"],
    ).to_csv(paths.artifacts_dir() / "image_manifest.csv", index=False)
    return tmp_path


@pytest.fixture
def make_client(
    root: Path,
    items: pd.DataFrame,
    make_llm: Callable[..., tuple[LLMClient, Any]],
) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(handler: Callable[[list[dict[str, str]]], str] = _handler) -> TestClient:
        llm, _ = make_llm(handler)
        backend = DemoBackend(
            items=items,
            queries=pd.DataFrame({"query_id": ["q001", "q002"], "text": [QUERY, "Shampoo"]}),
            systems=SYSTEMS,
            images=ImageIndex.default(),
            query_types={"q001": "occasion", "q002": "product"},
            llm=llm,
        )
        client = TestClient(create_app(backend))
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def test_queries_lists_eval_queries_and_demo_systems(make_client: Any) -> None:
    body = make_client().get("/api/queries").json()
    assert body["queries"][0] == {"query_id": "q001", "text": QUERY}
    assert [s["id"] for s in body["systems"]] == ["bm25", "main"]
    assert body["systems"][1] == {
        "id": "main",
        "label": "Main",
        "short": "Main",
        "desc": "Main pipeline",
    }
    assert body["judge_model"] == "gpt-4.1"


def test_eval_query_on_bm25_comes_from_the_run_without_trace(make_client: Any) -> None:
    body = make_client().get("/api/search", params={"q": QUERY, "system": "bm25"}).json()
    assert body["query_id"] == "q001" and body["query_type"] == "occasion"
    assert body["intent"] is None and body["food_prior"] is None and body["stages"] == []
    assert [r["item_id"] for r in body["results"]] == ["pizza", "burger", "fries"]
    first = body["results"][0]
    assert first["rank"] == 1 and first["score"] == 3.0
    assert first["sources"] == [{"label": "bm25", "rank": 1}]
    assert first["judge_grade"] is None and first["judge_reason"] is None
    assert first["doc_text"] == "Pizza de calabresa."
    assert body["metrics"] is None


def test_eval_query_on_pipeline_uses_run_meta_and_cached_trace(make_client: Any) -> None:
    body = make_client().get("/api/search", params={"q": QUERY, "system": "main"}).json()
    assert [r["item_id"] for r in body["results"]] == ["burger", "fries"]
    assert [s["name"] for s in body["stages"]] == ["query understanding", "listwise rerank"]
    assert body["stages"][0]["ms"] == 450.0
    assert body["intent"] == "occasion"
    assert body["expanded_dishes"] == ["hambúrguer", "batata frita"]
    assert body["understanding_model"] == "gpt-4.1-mini"
    assert body["food_prior"]["on"] is True and body["food_prior"]["lambda"] == 0.5
    labels = {s["label"] for s in body["results"][0]["sources"]}
    assert {"dense raw", "dense expanded"} <= labels


def test_free_text_query_runs_live(make_client: Any) -> None:
    body = make_client().get("/api/search", params={"q": "Hambúrguer", "system": "main"}).json()
    assert body["query_id"] is None and body["query_type"] is None and body["metrics"] is None
    assert body["intent"] == "occasion"
    assert "fusion" in [s["name"] for s in body["stages"]]
    assert len(body["results"]) == 4


def test_without_api_key_live_is_503_but_eval_queries_still_work(make_client: Any) -> None:
    client = make_client(_no_key)
    response = client.get("/api/search", params={"q": "Hambúrguer", "system": "main"})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
    body = client.get("/api/search", params={"q": QUERY, "system": "main"}).json()
    assert [r["item_id"] for r in body["results"]] == ["burger", "fries"]
    assert body["intent"] is None and len(body["stages"]) == 2
    assert client.get("/api/search", params={"q": "pizza", "system": "bm25"}).status_code == 200


def test_unknown_system_and_empty_query(make_client: Any) -> None:
    client = make_client()
    assert client.get("/api/search", params={"q": QUERY, "system": "nope"}).status_code == 404
    assert client.get("/api/search", params={"q": "", "system": "bm25"}).status_code == 422


def test_compare_takes_intent_from_the_side_with_understanding(make_client: Any) -> None:
    params = {"q": QUERY, "left": "bm25", "right": "main"}
    body = make_client().get("/api/compare", params=params).json()
    assert body["intent"] == "occasion" and body["query_type"] == "occasion"
    assert body["left"]["system"] == "bm25" and len(body["left"]["results"]) == 3
    assert body["right"]["system"] == "main" and body["right"]["metrics"] is None


def test_metrics_and_summary_read_reports_when_present(make_client: Any) -> None:
    reports = paths.reports_dir()
    reports.mkdir()
    pd.DataFrame(
        [
            {"system": "bm25", "query_id": "q001", "ndcg5": 0.5, "ndcg10": 0.6, "p5": 0.4},
            {"system": "main", "query_id": "q001", "ndcg5": 0.9, "ndcg10": None, "p5": 0.8},
        ]
    ).assign(food_leak5=0.0).to_csv(reports / "per_query.csv", index=False)
    pd.DataFrame(
        [
            {"system": "main", "metric": "ndcg5", "mean": 0.9, "lo": 0.8, "hi": 0.95},
            {"system": "main", "metric": "p5", "mean": 0.8, "lo": 0.7, "hi": 0.9},
        ]
    ).to_csv(reports / "metrics.csv", index=False)
    client = make_client()
    body = client.get("/api/search", params={"q": QUERY, "system": "bm25"}).json()
    assert body["metrics"]["main"] == {"ndcg5": 0.9, "ndcg10": None, "p5": 0.8, "food_leak5": 0.0}
    summary = client.get("/api/summary").json()
    assert summary["rows"] == [
        {
            "system": "main",
            "ndcg5": [0.9, 0.8, 0.95],
            "ndcg10": None,
            "p5": [0.8, 0.7, 0.9],
            "food_leak5": None,
        }
    ]


def test_summary_before_any_evaluation(make_client: Any) -> None:
    body = make_client().get("/api/summary").json()
    assert body["n_queries"] == 2 and body["rows"] == []
    assert body["judge"] == {"model": "gpt-4.1", "kappa": None, "judged": 0, "unjudged": 0}
    assert body["cost_usd"] is None
    assert body["label"] == {"done": 0, "total": 0}


def test_label_flow(make_client: Any) -> None:
    client = make_client()
    assert client.get("/api/label/queue").status_code == 404
    queue = [LabelPair("q001", "burger", 1, 1), LabelPair("q002", "shampoo", 1, 2)]
    save_queue([*queue, LabelPair("q001", "burger", 2, 1)], paths.labels_dir() / "queue.csv")

    body = client.get("/api/label/queue").json()
    assert (body["done"], body["total"], len(body["pairs"])) == (0, 3, 3)
    pair = body["pairs"][0]
    assert pair["pair_id"] == "q001:burger:1" and pair["query"] == QUERY
    assert set(pair["item"]) == {
        "item_id",
        "name",
        "category_path",
        "description",
        "price",
        "price_bucket",
        "attributes",
        "is_food",
    }

    assert client.post("/api/label", json={"pair_id": "q001:burger:1", "grade": 3}).json() == {
        "done": 1,
        "total": 3,
    }
    assert client.post("/api/label", json={"pair_id": "q001:burger:1", "grade": 2}).json() == {
        "done": 1,
        "total": 3,
    }
    assert (
        client.post("/api/label", json={"pair_id": "q001:burger:1", "grade": 4}).status_code == 422
    )
    assert client.post("/api/label", json={"pair_id": "q009:x:1", "grade": 1}).status_code == 404
    remaining = client.get("/api/label/queue").json()["pairs"]
    assert [p["pair_id"] for p in remaining] == ["q002:shampoo:1", "q001:burger:2"]

    export = client.get("/api/label/export")
    assert export.status_code == 200 and export.headers["content-type"].startswith("text/csv")
    rows = pd.read_csv(io.StringIO(export.text)).to_dict("records")
    assert len(rows) == 1 and rows[0]["grade"] == 2 and rows[0]["pass"] == 1


def test_label_export_404_before_labels(make_client: Any) -> None:
    assert make_client().get("/api/label/export").status_code == 404


def test_images_are_served_from_disk_with_sniffed_type(make_client: Any) -> None:
    client = make_client()
    response = client.get("/images/burger")
    assert response.status_code == 200 and response.headers["content-type"] == "image/png"
    assert response.content == PNG
    assert client.get("/images/pizza").status_code == 404
    assert client.get("/images/unknown").status_code == 404


def test_static_frontend_is_served(make_client: Any) -> None:
    client = make_client()
    index = client.get("/")
    assert index.status_code == 200 and "fs-root" in index.text
    assert client.get("/js/api.js").status_code == 200


def test_image_report_warns_when_cache_is_missing(
    make_client: Any, items: pd.DataFrame, root: Path
) -> None:
    backend: DemoBackend = make_client().app.state.backend
    ok, report = backend.image_report()
    assert ok and report.startswith("images: 1 / 4 items")
    backend.images = ImageIndex({})
    ok, report = backend.image_report()
    assert not ok and "fetch_images.py" in report and "artifacts/" in report


def test_queries_fall_back_to_query_types_without_data(root: Path) -> None:
    pd.DataFrame(
        {"query_id": ["q001"], "search_term_pt": [f" {QUERY} "], "type": ["occasion"]}
    ).to_csv(paths.query_types_csv(), index=False)
    assert load_demo_queries().to_dict("records") == [{"query_id": "q001", "text": QUERY}]


def test_frontend_files_are_revalidated_but_api_is_not(make_client: Any) -> None:
    client = make_client()
    assert client.get("/js/format.js").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get("/api/queries").headers
