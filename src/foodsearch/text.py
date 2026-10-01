import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

from unidecode import unidecode

if TYPE_CHECKING:
    from nltk.stem import SnowballStemmer

_WORD = re.compile(r"\w+")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    import nltk

    try:
        words = nltk.corpus.stopwords.words("portuguese")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("portuguese")
    return frozenset(unidecode(w) for w in words)


@lru_cache(maxsize=1)
def _stemmer() -> "SnowballStemmer":
    from nltk.stem import SnowballStemmer

    return SnowballStemmer("portuguese")


def tokenize(text: str) -> list[str]:
    stopwords, stemmer = _stopwords(), _stemmer()
    words = _WORD.findall(unicodedata.normalize("NFKC", text).lower())
    return [
        unidecode(stemmer.stem(w)) for w in words if len(w) > 1 and unidecode(w) not in stopwords
    ]
