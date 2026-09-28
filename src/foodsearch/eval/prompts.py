import json
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from foodsearch.cards import item_card
from foodsearch.llm import text_hash

RUBRIC_VERSION = "v1"
Lang = Literal["en", "pt"]
LANGS: tuple[Lang, ...] = ("en", "pt")

SYSTEM_PROMPT: dict[Lang, str] = {
    "en": """You are a relevance assessor for a Brazilian food-delivery search engine.
You receive a customer search query in Portuguese and one catalog item.
Grade how well the item satisfies the query on a 0-3 scale:

3 perfect: matches the core intent and its key modifiers (style, cuisine,
  meal time, dietary need). The customer would order it for this query.
2 good: right dish or right kind of food, but misses or only partly meets a
  modifier. A reasonable substitute.
1 partial: related but not what was asked: an ingredient, a side, a
  grocery/frozen/raw version of the dish, a drink that fits the occasion,
  or the right cuisine but the wrong meal type.
0 irrelevant: unrelated food, or any non-food item for a food query.

Rules:
- For a dish query, a grocery, frozen, raw or ingredient version is at most 1,
  unless the query asks for groceries or ingredients.
- Right dish with a missing or wrong modifier is 2. Right dish with the
  modifier clearly contradicted is 1.
- Occasion queries: a fitting main dish is 3; a side, snack or drink that fits
  well on its own is 2; one that only accompanies a meal is 1.
- Cuisine or style queries: a typical dish of that cuisine is 3; a generic dish
  that could fit is 2; only a shared ingredient is 1.
- Any non-food item (hygiene, pharmacy, pet, home, electronics) for a food
  query is 0, regardless of word overlap.
- Product queries: same product is 3; same product type from another brand or
  variant is 2; same broad use is 1.
- Portion size and price do not change the grade unless the query states them.
- Judge only what the item text supports. Do not guess what a vague item
  might be.
- Queries may be literal translations of English food names (e.g. a "club
  sandwich" written word by word). Grade against the dish the customer means.

Answer with JSON only: {"grade": <0-3>, "reason": "<max 25 words>"}""",
    "pt": """Você é um avaliador de relevância para um buscador de delivery de comida no Brasil.
Você recebe uma busca de um cliente em português e um item do catálogo.
Avalie o quanto o item atende à busca numa escala de 0 a 3:

3 perfeito: atende à intenção principal e aos modificadores-chave (estilo,
  culinária, refeição do dia, restrição alimentar). O cliente pediria este
  item para esta busca.
2 bom: prato certo ou tipo certo de comida, mas não atende ou atende só em
  parte a um modificador. Um substituto razoável.
1 parcial: relacionado, mas não é o que foi pedido: um ingrediente, um
  acompanhamento, uma versão de mercado/congelada/crua do prato, uma bebida
  que combina com a ocasião, ou a culinária certa com o tipo de refeição errado.
0 irrelevante: comida sem relação, ou qualquer item que não é comida para uma
  busca de comida.

Regras:
- Para uma busca de prato, uma versão de mercado, congelada, crua ou de
  ingrediente vale no máximo 1, a menos que a busca peça mercado ou ingredientes.
- Prato certo com modificador ausente ou errado vale 2. Prato certo com o
  modificador claramente contrariado vale 1.
- Buscas de ocasião: um prato principal adequado vale 3; um acompanhamento,
  petisco ou bebida que funciona bem sozinho vale 2; um que só acompanha uma
  refeição vale 1.
- Buscas de culinária ou estilo: um prato típico dessa culinária vale 3; um
  prato genérico que poderia se encaixar vale 2; só um ingrediente em comum vale 1.
- Qualquer item que não é comida (higiene, farmácia, pet, casa, eletrônicos)
  para uma busca de comida vale 0, mesmo que as palavras coincidam.
- Buscas de produto: o mesmo produto vale 3; o mesmo tipo de produto de outra
  marca ou variante vale 2; o mesmo uso geral vale 1.
- Tamanho da porção e preço não mudam a nota, a menos que a busca os mencione.
- Avalie só o que o texto do item sustenta. Não adivinhe o que um item vago
  poderia ser.
- As buscas podem ser traduções literais de nomes de comida em inglês (por
  exemplo, "club sandwich" traduzido palavra por palavra). Avalie pelo prato
  que o cliente quer dizer.

Responda só com JSON: {"grade": <0-3>, "reason": "<máx. 25 palavras>"}""",
}


@dataclass(frozen=True, slots=True)
class FewShot:
    query: str
    item: dict[str, object]
    grade: int
    reason: dict[Lang, str]


def _item(
    name: str, category: str, l1: str, l2: str, description: str, price: float
) -> dict[str, object]:
    return {
        "name": name,
        "category_name": category,
        "l1": l1,
        "l2": l2,
        "description_clean": description,
        "price": price,
        "vegan": False,
        "lac_free": False,
        "organic": False,
        "tags": {},
    }


FEW_SHOTS: tuple[FewShot, ...] = (
    FewShot(
        query="Algo quente para uma noite fria",
        item=_item(
            "Caldo verde com linguiça 500ml",
            "Sopas e Caldos",
            "PRATOS",
            "SOPAS",
            "Caldo verde cremoso com couve e rodelas de linguiça calabresa.",
            24.90,
        ),
        grade=3,
        reason={
            "en": "Hot soup, exactly what a cold night calls for.",
            "pt": "Sopa quente, exatamente o que uma noite fria pede.",
        },
    ),
    FewShot(
        query="Jantar leve estilo italiano",
        item=_item(
            "Lasanha à bolonhesa tamanho família",
            "Massas",
            "PRATOS",
            "MASSAS",
            "Lasanha com molho bolonhesa, presunto e muçarela gratinada. Serve 4 pessoas.",
            79.90,
        ),
        grade=2,
        reason={
            "en": "Italian dish, but heavy and large, not a light dinner.",
            "pt": "Prato italiano, mas pesado e grande, não é um jantar leve.",
        },
    ),
    FewShot(
        query="Pastel de feira de carne",
        item=_item(
            "Massa para pastel em rolo 500g",
            "Mercearia",
            "MASSAS_FRESCAS",
            "MASSA_PASTEL",
            "",
            9.49,
        ),
        grade=1,
        reason={
            "en": "Raw dough to make pastel, not a ready pastel.",
            "pt": "Massa crua para fazer pastel, não um pastel pronto.",
        },
    ),
    FewShot(
        query="Petisco para assistir futebol",
        item=_item(
            "Carregador USB-C 20W",
            "Eletrônicos",
            "ACESSORIOS",
            "CARREGADORES",
            "Carregador de parede com cabo USB-C.",
            59.90,
        ),
        grade=0,
        reason={"en": "Electronics, not food.", "pt": "Eletrônico, não é comida."},
    ),
    FewShot(
        query="Detergente líquido neutro 500ml",
        item=_item(
            "Detergente líquido neutro 500ml marca própria",
            "Limpeza",
            "LIMPEZA_COZINHA",
            "DETERGENTE",
            "",
            2.79,
        ),
        grade=3,
        reason={
            "en": "Same product type, variant and size.",
            "pt": "Mesmo tipo de produto, variante e tamanho.",
        },
    ),
)


def user_message(query: str, card: str) -> str:
    return f"Consulta: {query}\n\nItem:\n{card}"


def few_shot_messages(lang: Lang) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for shot in FEW_SHOTS:
        answer = {"grade": shot.grade, "reason": shot.reason[lang]}
        messages += [
            {"role": "user", "content": user_message(shot.query, item_card(pd.Series(shot.item)))},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
        ]
    return messages


def judge_messages(query: str, card: str, lang: Lang) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT[lang]},
        *few_shot_messages(lang),
        {"role": "user", "content": user_message(query, card)},
    ]


def system_prompt_hash(lang: Lang) -> str:
    return text_hash(SYSTEM_PROMPT[lang])


def few_shot_hash(lang: Lang) -> str:
    return text_hash(json.dumps(few_shot_messages(lang), ensure_ascii=False, sort_keys=True))
