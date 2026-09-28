import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SHIFTS = ("breakfast", "lunch", "snack", "dinner", "dawn")
EXPECTED_ITEMS = 4997

PACKAGING = (
    r"embalagem|unidade|lata|caixa|garrafa|pacote|pote|frasco|bisnaga|tubo|refil|saco|sach[eê]"
    r"|long neck|vidro|fardo|kit|bandeja|maço|bl[ií]ster|cartela"
)
BOILERPLATE = (
    re.compile(r"^produto para maiores de 18 anos\.?$", re.I),
    re.compile(r"^compra por peso\.?$", re.I),
    re.compile(r"^imagem ilustrativa", re.I),
    re.compile(rf"^(?:(?:{PACKAGING})\s*)?(?:c/|com)?\s*[\d.,]+\s*[a-zà-ú]{{0,12}}\.?$", re.I),
)
MAX_DESCRIPTION = 300

PRICE_LABELS = ("barato", "médio", "caro")

ATTR_PT = {
    "VEGAN": "vegano",
    "VEGETARIAN": "vegetariano",
    "LAC_FREE": "sem lactose",
    "GLUTEN_FREE": "sem glúten",
    "SUGAR_FREE": "sem açúcar",
    "ORGANIC": "orgânico",
    "NATURAL": "natural",
    "DIET": "diet",
    "ZERO": "zero",
    "ALCOHOLIC_DRINK": "bebida alcoólica",
    "FROSTY": "gelado",
}

FOOD_L0 = frozenset(
    {
        "ALIMENTOS_PREPARADOS",
        "MERCEARIA",
        "BEBIDAS",
        "BOMBONIERE",
        "FRIOS_LATICINIOS",
        "FLV",
        "PROTEINAS",
    }
)
NON_FOOD_OUTROS = frozenset(
    {
        "Farmácia Veterinária",
        "Gatos",
        "Higiene e Beleza",
        "Higiene e Limpeza",
        "Limpeza",
        "Mamãe e Bebê",
        "Medicamentos",
        "Papéis",
        "Primeiros Socorros",
        "Roedores",
        "Saúde e Bem-Estar",
        "Suplementos",
        "Suplementos e Vitaminas",
    }
)

JSON_COLUMNS = ("tags", "search_terms")


def _parse_row(item_id: str, merchant_id: str, meta_s: str, prof_s: str) -> dict[str, Any]:
    meta, prof = json.loads(meta_s), json.loads(prof_s)
    tax, metrics = meta.get("taxonomy") or {}, prof.get("metrics") or {}
    shift = (metrics.get("orderingRate") or {}).get("shift") or {}
    return {
        "item_id": item_id,
        "merchant_id": merchant_id,
        "name": (meta.get("name") or "").strip(),
        "category_name": (meta.get("category_name") or "").strip(),
        "description": (meta.get("description") or "").strip(),
        "price": float(meta["price"]) if meta.get("price") is not None else np.nan,
        "l0": tax.get("l0"),
        "l1": tax.get("l1"),
        "l2": tax.get("l2"),
        "vegan": bool(meta.get("vegan")),
        "lac_free": bool(meta.get("lacFree")),
        "organic": bool(meta.get("organic")),
        "tags": {t["key"]: list(t["value"]) for t in meta.get("tags") or []},
        "images": list(meta.get("images") or []),
        "search_terms": [
            {
                "term": t["term"],
                "count": int(t.get("count") or 0),
                "method": t.get("method") or "unknown",
            }
            for t in prof.get("search") or []
        ],
        "total_orders": int(metrics.get("total_orders") or 0),
        "reorder_rate": float(metrics.get("reorderRate") or 0.0),
        "conversion_rate": float(metrics.get("conversionRate") or 0.0),
        **{f"shift_{s}": float(shift.get(s) or 0.0) for s in SHIFTS},
    }


def parse_items(raw: pd.DataFrame) -> pd.DataFrame:
    columns = raw[["itemId", "merchantId", "itemMetadata", "itemProfile"]]
    return pd.DataFrame([_parse_row(*row) for row in columns.itertuples(index=False)])


def dedupe(raw: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    dup_ids = raw.loc[raw["itemId"].duplicated(keep=False), "itemId"].unique()
    for item_id in dup_ids:
        rows = raw.loc[raw["itemId"] == item_id].drop(columns="_id", errors="ignore")
        if rows.nunique().max() != 1:
            raise ValueError(f"Duplicated itemId {item_id} has differing rows")
    return items.drop_duplicates("item_id").reset_index(drop=True)


def clean_description(text: str, name: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if any(p.match(text) for p in BOILERPLATE) or text.lower().rstrip(".") in name.lower():
        return ""
    if len(text) <= MAX_DESCRIPTION:
        return text
    return text[:MAX_DESCRIPTION].rsplit(" ", 1)[0] + "..."


def humanize(code: object) -> str:
    return code.replace("_", " ").lower().strip() if isinstance(code, str) else ""


def price_buckets(items: pd.DataFrame) -> pd.Series:
    ranks = items.groupby("l0")["price"].rank(pct=True)
    buckets = pd.cut(ranks, [0, 1 / 3, 2 / 3, 1], labels=list(PRICE_LABELS), include_lowest=True)
    return buckets.astype(str)


def item_attributes(row: pd.Series) -> list[str]:
    flags = [("VEGAN", row["vegan"]), ("LAC_FREE", row["lac_free"]), ("ORGANIC", row["organic"])]
    codes = [code for code, flag in flags if flag]
    for key in ("DIETARY_RESTRICTIONS", "DISH_CLASSIFICATION"):
        codes += row["tags"].get(key, [])
    return list(dict.fromkeys(ATTR_PT.get(c, humanize(c)) for c in codes))


def build_doc(row: pd.Series) -> str:
    path = [row["category_name"], humanize(row["l1"]), humanize(row["l2"])]
    category = " > ".join(dict.fromkeys(p for p in path if p))
    attrs = ", ".join(item_attributes(row))
    parts = [
        row["name"],
        f"Categoria: {category}" if category else "",
        row["description_clean"],
        attrs.capitalize() if attrs else "",
        f"Preço: {row['price_bucket']}",
    ]
    return ". ".join(p.rstrip(".") for p in parts if p) + "."


def is_food(l0: str | None, category_name: str) -> bool:
    return l0 in FOOD_L0 or (l0 == "OUTROS" and category_name not in NON_FOOD_OUTROS)


def prepare_items(raw: pd.DataFrame) -> pd.DataFrame:
    items = dedupe(raw, parse_items(raw))
    items["description_clean"] = [
        clean_description(d, n) for d, n in zip(items["description"], items["name"], strict=True)
    ]
    items["price_bucket"] = price_buckets(items)
    items["doc"] = items.apply(build_doc, axis=1)
    items["is_food"] = [
        is_food(l0, c) for l0, c in zip(items["l0"], items["category_name"], strict=True)
    ]
    items["is_dish"] = items["l0"] == "ALIMENTOS_PREPARADOS"
    return items


def build_items(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, dtype={"itemId": str, "merchantId": str})
    items = prepare_items(raw)
    if len(items) != EXPECTED_ITEMS:
        raise ValueError(f"Expected {EXPECTED_ITEMS} items after dedupe, got {len(items)}")
    return items


def save_items(items: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = items.drop(columns=list(JSON_COLUMNS)).assign(
        **{c: items[c].map(lambda v: json.dumps(v, ensure_ascii=False)) for c in JSON_COLUMNS}
    )
    out.to_parquet(path, index=False)


def load_items(path: Path) -> pd.DataFrame:
    items = pd.read_parquet(path)
    for column in JSON_COLUMNS:
        items[column] = items[column].map(json.loads)
    items["images"] = items["images"].map(list)
    return items


def load_queries(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    return pd.DataFrame(
        {
            "query_id": [f"q{i:03d}" for i in range(1, len(raw) + 1)],
            "text": raw["search_term_pt"].str.strip(),
        }
    )
