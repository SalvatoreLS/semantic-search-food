"""Item cards: the single text view of a catalog item.

The LLM judge, the human Label view and the LLM rerankers all read the same card: name, category
path, cleaned description, price and attributes. Graders and rerankers therefore see exactly the
same evidence, which keeps the judge-vs-human agreement check fair and keeps the reranker from using
information the judge cannot see.
"""

from typing import Any

import pandas as pd

from foodsearch.data import humanize, item_attributes


def category_path(row: pd.Series, sep: str = " > ") -> str:
    parts = [row["category_name"], humanize(row["l1"]), humanize(row["l2"])]
    return sep.join(dict.fromkeys(p for p in parts if p))


def price_text(price: float) -> str:
    return f"R$ {price:.2f}" if price > 0 else "(não informado)"


def item_card(row: pd.Series) -> str:
    attributes = ", ".join(item_attributes(row)) or "(nenhum)"
    return "\n".join(
        [
            f"Nome: {row['name']}",
            f"Categoria: {category_path(row) or '(sem categoria)'}",
            f"Descrição: {row['description_clean'] or '(sem descrição)'}",
            f"Preço: {price_text(row['price'])}",
            f"Atributos: {attributes}",
        ]
    )


def card_fields(row: pd.Series) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "name": row["name"],
        "category_path": category_path(row, sep=" \u203a "),
        "description": row["description_clean"],
        "price": float(row["price"]),
        "price_bucket": row["price_bucket"],
        "attributes": item_attributes(row),
        "is_food": bool(row["is_food"]),
    }
