import pytest

from foodsearch.text import tokenize


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Sanduíches", "sanduiche"),
        ("Pães", "pão"),
        ("hambúrgueres", "Hambúrguer"),
        ("Açaí", "acai"),
        ("caseira", "caseiro"),
    ],
)
def test_inflections_and_accents_match(a: str, b: str) -> None:
    assert tokenize(a) == tokenize(b)


def test_drops_stopwords_and_single_characters() -> None:
    assert tokenize("Almoço de domingo com a família e") == tokenize("almoço domingo família")


def test_output_is_accent_free() -> None:
    assert all(t.isascii() for t in tokenize("Feijão tropeiro à mineira com açúcar"))


def test_empty_text() -> None:
    assert tokenize("") == []
    assert tokenize("de a o") == []
