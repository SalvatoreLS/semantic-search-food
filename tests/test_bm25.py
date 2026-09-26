import pandas as pd
import pytest

from foodsearch.retrievers import BM25Retriever, Hit


@pytest.fixture
def retriever() -> BM25Retriever:
    items = pd.DataFrame(
        {
            "item_id": ["burger", "pizza", "shampoo", "xburger"],
            "doc": [
                "Hambúrguer artesanal com queijo. Categoria: Lanches.",
                "Pizza de calabresa. Categoria: Pizzas.",
                "Shampoo neutro. Categoria: Higiene.",
                "X-Burger com bacon e queijo. Categoria: Lanches.",
            ],
        }
    )
    r = BM25Retriever()
    r.fit(items)
    return r


def test_ranks_matching_items_first(retriever: BM25Retriever) -> None:
    hits = retriever.search("hambúrgueres artesanais")
    assert hits[0] == Hit(item_id="burger", score=hits[0].score, source="bm25")
    assert all(h.source == "bm25" for h in hits)


def test_drops_zero_score_hits(retriever: BM25Retriever) -> None:
    hits = retriever.search("queijo", k=10)
    assert {h.item_id for h in hits} == {"burger", "xburger"}
    assert all(h.score > 0 for h in hits)


def test_scores_are_descending_and_k_is_respected(retriever: BM25Retriever) -> None:
    hits = retriever.search("lanches queijo pizza", k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_unknown_or_stopword_query_returns_nothing(retriever: BM25Retriever) -> None:
    assert retriever.search("sushi") == []
    assert retriever.search("de com a") == []


def test_search_before_fit_fails() -> None:
    with pytest.raises(RuntimeError, match="fitted"):
        BM25Retriever().search("pizza")
