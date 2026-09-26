import json
from pathlib import Path

import pandas as pd
import pytest

from foodsearch.data import (
    MAX_DESCRIPTION,
    build_doc,
    clean_description,
    dedupe,
    humanize,
    is_food,
    load_items,
    load_queries,
    parse_items,
    prepare_items,
    price_buckets,
    save_items,
)


def _raw_row(item_id: str, name: str, l0: str, price: float, **meta: object) -> dict[str, str]:
    metadata = {
        "name": name,
        "category_name": meta.pop("category_name", "Lanches"),
        "description": meta.pop("description", ""),
        "price": price,
        "taxonomy": {"l0": l0, "l1": meta.pop("l1", None), "l2": None},
        "tags": meta.pop("tags", []),
        "images": ["abc.jpg"],
        **meta,
    }
    profile = {"search": [{"term": "lanche", "count": 2, "method": "term"}], "metrics": {}}
    return {
        "_id": f"mongo-{item_id}",
        "itemId": item_id,
        "merchantId": "m1",
        "itemMetadata": json.dumps(metadata),
        "itemProfile": json.dumps(profile),
    }


@pytest.fixture
def raw() -> pd.DataFrame:
    rows = [
        _raw_row("a", "X-Burger", "ALIMENTOS_PREPARADOS", 20.0, l1="SANDUICHES", vegan=True),
        _raw_row("b", "Shampoo Neutro", "BELEZA_ESTETICA_HIGIENE_PESSOAL", 15.0),
        _raw_row("c", "Açaí 500ml", "ALIMENTOS_PREPARADOS", 30.0, description="Embalagem 500ml"),
    ]
    rows.append({**rows[0], "_id": "mongo-dup"})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "text",
    ["Embalagem 300g", "Compra por peso", "Produto para maiores de 18 anos.", "500 ml", "X-Burger"],
)
def test_clean_description_blanks_boilerplate_and_name_repeats(text: str) -> None:
    assert clean_description(text, "X-Burger") == ""


def test_clean_description_collapses_whitespace_and_truncates() -> None:
    assert clean_description("Pão,  carne\n e queijo", "X") == "Pão, carne e queijo"
    long = " ".join(["palavra"] * 100)
    cleaned = clean_description(long, "X")
    assert cleaned.endswith("...") and len(cleaned) <= MAX_DESCRIPTION + 3


def test_humanize() -> None:
    assert humanize("SORVETES_ACAIS_FRUTAS") == "sorvetes acais frutas"
    assert humanize(None) == ""


def test_price_buckets_are_terciles_within_l0() -> None:
    items = pd.DataFrame({"l0": ["A"] * 3 + ["B"] * 3, "price": [1, 2, 3, 100, 200, 300]})
    assert price_buckets(items).tolist() == ["barato", "médio", "caro"] * 2


def test_is_food_uses_l0_and_outros_category() -> None:
    assert is_food("MERCEARIA", "Qualquer")
    assert not is_food("PET", "Ração")
    assert is_food("OUTROS", "Doces")
    assert not is_food("OUTROS", "Limpeza")


def test_dedupe_keeps_one_of_identical_rows(raw: pd.DataFrame) -> None:
    items = dedupe(raw, parse_items(raw))
    assert items["item_id"].tolist() == ["a", "b", "c"]


def test_dedupe_rejects_differing_duplicates(raw: pd.DataFrame) -> None:
    raw.loc[3, "merchantId"] = "m2"
    with pytest.raises(ValueError, match="differing rows"):
        dedupe(raw, parse_items(raw))


def test_prepare_items_builds_doc_and_flags(raw: pd.DataFrame) -> None:
    items = prepare_items(raw).set_index("item_id")
    assert items.loc["a", "doc"] == (
        "X-Burger. Categoria: Lanches > sanduiches. Vegano. Preço: médio."
    )
    assert items.loc["c", "description_clean"] == ""
    assert items["is_food"].to_dict() == {"a": True, "b": False, "c": True}
    assert items["is_dish"].to_dict() == {"a": True, "b": False, "c": True}


def test_build_doc_joins_attributes_and_dedupes_category_path() -> None:
    row = pd.Series(
        {
            "name": "Salada.",
            "category_name": "saladas",
            "l1": "SALADAS",
            "l2": None,
            "description_clean": "Folhas verdes",
            "vegan": True,
            "lac_free": True,
            "organic": False,
            "tags": {"DIETARY_RESTRICTIONS": ["VEGAN", "GLUTEN_FREE"]},
            "price_bucket": "caro",
        }
    )
    assert build_doc(row) == (
        "Salada. Categoria: saladas. Folhas verdes. Vegano, sem lactose, sem glúten. Preço: caro."
    )


def test_items_parquet_roundtrip(raw: pd.DataFrame, tmp_path: Path) -> None:
    items = prepare_items(raw)
    save_items(items, tmp_path / "items.parquet")
    back = load_items(tmp_path / "items.parquet")
    assert back["doc"].equals(items["doc"])
    assert back["tags"].tolist() == items["tags"].tolist()
    assert back["search_terms"].tolist() == items["search_terms"].tolist()
    assert back["images"].tolist() == items["images"].tolist()


def test_query_ids_follow_row_order(tmp_path: Path) -> None:
    path = tmp_path / "queries.csv"
    pd.DataFrame({"search_term_pt": ["Pizza ", "Sushi"]}).to_csv(path, index=False)
    queries = load_queries(path)
    assert queries["query_id"].tolist() == ["q001", "q002"]
    assert queries["text"].tolist() == ["Pizza", "Sushi"]
