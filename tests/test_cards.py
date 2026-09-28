import pandas as pd

from foodsearch.cards import card_fields, item_card


def _row(**overrides: object) -> pd.Series:
    row = {
        "item_id": "i1",
        "name": "Pizza margherita",
        "category_name": "Pizzas",
        "l1": "PIZZAS",
        "l2": "PIZZA_SALGADA",
        "description_clean": "Molho de tomate, muçarela e manjericão.",
        "price": 42.5,
        "price_bucket": "médio",
        "vegan": False,
        "lac_free": False,
        "organic": False,
        "tags": {"DIETARY_RESTRICTIONS": ["VEGETARIAN"]},
        "is_food": True,
    }
    return pd.Series({**row, **overrides})


def test_card_renders_all_fields() -> None:
    assert item_card(_row()) == (
        "Nome: Pizza margherita\n"
        "Categoria: Pizzas > pizzas > pizza salgada\n"
        "Descrição: Molho de tomate, muçarela e manjericão.\n"
        "Preço: R$ 42.50\n"
        "Atributos: vegetariano"
    )


def test_card_placeholders_for_missing_fields() -> None:
    card = item_card(_row(description_clean="", tags={}, price=0.0, l1=None, l2=None))
    assert "Categoria: Pizzas\n" in card
    assert "Descrição: (sem descrição)" in card
    assert "Preço: (não informado)" in card
    assert card.endswith("Atributos: (nenhum)")


def test_card_fields_have_no_ranking_information() -> None:
    fields = card_fields(_row())
    assert set(fields) == {
        "item_id",
        "name",
        "category_path",
        "description",
        "price",
        "price_bucket",
        "attributes",
        "is_food",
    }
    assert fields["category_path"] == "Pizzas \u203a pizzas \u203a pizza salgada"
    assert fields["attributes"] == ["vegetariano"]
