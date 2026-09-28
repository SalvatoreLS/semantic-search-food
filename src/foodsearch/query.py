import json
from functools import cache
from typing import Literal

from pydantic import BaseModel, Field

from foodsearch.llm import LLMClient
from foodsearch.text import tokenize

UNDERSTANDING_MODEL = "gpt-4.1-mini"

Intent = Literal["dish", "grocery", "product", "occasion"]
MealShift = Literal["breakfast", "lunch", "snack", "dinner", "dawn", "any"]


class QueryUnderstanding(BaseModel):
    intent: Intent
    dishes_pt: list[str] = Field(min_length=1, max_length=8)
    keywords_pt: list[str] = Field(max_length=8)
    meal_shift: MealShift
    dietary: list[str] = Field(max_length=5)

    def expanded_text(self, query: str) -> str:
        return "; ".join(dict.fromkeys([query, *self.dishes_pt]))


SYSTEM_PROMPT = """\
You interpret search queries typed into a Brazilian food-delivery app whose catalog mixes \
restaurant dishes, groceries, drinks, pharmacy, hygiene, pet and home products. \
Queries are in Portuguese and are often abstract (an occasion, a cuisine, a style) or \
translated from US food concepts.

Return one JSON object with:
- "intent": "dish" (a prepared dish or dish style), "occasion" (a moment or situation that \
calls for food), "grocery" (packaged food or drink to buy), or "product" (a non-food product \
such as cleaning, hygiene, pharmacy or pet items).
- "dishes_pt": 3 to 8 concrete items, in Brazilian Portuguese, that a Brazilian delivery app \
would sell for this query, most typical first. For a "product" query, list the product itself \
and close variants. Use names as they appear on Brazilian menus, not literal translations.
- "keywords_pt": up to 8 short Portuguese keywords (ingredients, cuisine, style).
- "meal_shift": "breakfast", "lunch", "snack", "dinner", "dawn" or "any".
- "dietary": dietary constraints stated or clearly implied (e.g. "vegetariano"), else [].

Do not invent brands or restaurants. Keep the user's meaning; do not narrow it more than needed."""

FEW_SHOTS: tuple[tuple[str, dict[str, object]], ...] = (
    (
        "Lanche para levar na trilha",
        {
            "intent": "occasion",
            "dishes_pt": ["barra de cereal", "sanduíche natural", "mix de castanhas", "frutas"],
            "keywords_pt": ["prático", "leve", "energia"],
            "meal_shift": "snack",
            "dietary": [],
        },
    ),
    (
        "Moqueca baiana tradicional",
        {
            "intent": "dish",
            "dishes_pt": ["moqueca de peixe", "moqueca de camarão", "bobó de camarão", "pirão"],
            "keywords_pt": ["baiana", "dendê", "leite de coco", "frutos do mar"],
            "meal_shift": "lunch",
            "dietary": [],
        },
    ),
    (
        "Xarope para tosse infantil",
        {
            "intent": "product",
            "dishes_pt": ["xarope para tosse infantil", "xarope expectorante", "xarope de mel"],
            "keywords_pt": ["tosse", "infantil", "farmácia"],
            "meal_shift": "any",
            "dietary": [],
        },
    ),
)


def understanding_messages(query: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for example, answer in FEW_SHOTS:
        messages.append({"role": "user", "content": example})
        messages.append({"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)})
    messages.append({"role": "user", "content": query})
    return messages


def understand(query: str, llm: LLMClient, model: str = UNDERSTANDING_MODEL) -> QueryUnderstanding:
    return llm.chat_checked(
        model,
        understanding_messages(query),
        QueryUnderstanding.model_validate,
        tag="query",
        max_tokens=300,
    )


NON_FOOD_TERMS = (
    "desinfetante",
    "detergente",
    "sabão",
    "sabonete",
    "shampoo",
    "xampu",
    "condicionador",
    "desodorante",
    "amaciante",
    "alvejante",
    "água sanitária",
    "limpador",
    "multiuso",
    "inseticida",
    "repelente",
    "protetor solar",
    "creme dental",
    "escova de dente",
    "papel higiênico",
    "fralda",
    "absorvente",
    "preservativo",
    "ração",
    "areia para gato",
    "remédio",
    "medicamento",
    "comprimido",
    "pomada",
    "curativo",
    "perfume",
    "maquiagem",
    "esmalte",
)


@cache
def _non_food_stems() -> tuple[frozenset[str], ...]:
    return tuple(frozenset(tokenize(term)) for term in NON_FOOD_TERMS)


def rule_intent(query: str) -> Intent:
    tokens = set(tokenize(query))
    return "product" if any(stems <= tokens for stems in _non_food_stems()) else "dish"
