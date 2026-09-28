import re
import unicodedata
from functools import lru_cache

import nltk
from nltk.stem import SnowballStemmer
from unidecode import unidecode

_WORD = re.compile(r"\w+")


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    try:
        words = nltk.corpus.stopwords.words("portuguese")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk.corpus.stopwords.words("portuguese")
    return frozenset(unidecode(w) for w in words)


@lru_cache(maxsize=1)
def _stemmer() -> SnowballStemmer:
    return SnowballStemmer("portuguese")


def tokenize(text: str) -> list[str]:
    stopwords, stemmer = _stopwords(), _stemmer()
    words = _WORD.findall(unicodedata.normalize("NFKC", text).lower())
    return [
        unidecode(stemmer.stem(w)) for w in words if len(w) > 1 and unidecode(w) not in stopwords
    ]
