"""Text preprocessing: normalization, negation detection, Phrase builder."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_NEGATION_CUES: frozenset[str] = frozenset({"not", "no", "never", "without", "n't"})
_FAIL_VERBS: frozenset[str] = frozenset({"fail", "failed", "fails"})

_NLTK_PACKAGES: tuple[tuple[str, str], ...] = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("corpora/stopwords", "stopwords"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


@lru_cache(maxsize=1)
def ensure_nltk_data() -> None:
    """Download required NLTK corpora if absent."""
    for path, pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk_data()
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _active_stopwords() -> frozenset[str]:
    ensure_nltk_data()
    return frozenset(stopwords.words("english")) - _NEGATION_CUES


def _wordnet_pos(tag: str) -> str:
    """Map Penn Treebank POS tag to WordNet POS constant."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def _is_wordlike(token: str) -> bool:
    return any(ch.isalnum() for ch in token)


def normalize(text: str) -> list[str]:
    """Lowercase, tokenize, drop stopwords (keep negation cues), lemmatize by POS."""
    ensure_nltk_data()
    tagged = nltk.pos_tag(word_tokenize(text.lower()))
    stops = _active_stopwords()
    lemma = _lemmatizer()
    out: list[str] = []
    for token, tag in tagged:
        if not _is_wordlike(token):
            continue
        if token in stops:
            continue
        out.append(lemma.lemmatize(token, _wordnet_pos(tag)))
    return out


def detect_negation(text: str) -> bool:
    """True if text contains a negation cue or a 'fail(s|ed) to' pattern."""
    ensure_nltk_data()
    tokens = word_tokenize(text.lower())
    for i, token in enumerate(tokens):
        if token in _NEGATION_CUES:
            return True
        if token in _FAIL_VERBS and i + 1 < len(tokens) and tokens[i + 1] == "to":
            return True
    return False


@dataclass(frozen=True)
class Phrase:
    """Raw text plus its normalized tokens and a negation flag."""
    raw: str
    tokens: tuple[str, ...]
    has_negation: bool


def build_phrase(text: str) -> Phrase:
    """Build a Phrase from raw input text."""
    return Phrase(
        raw=text,
        tokens=tuple(normalize(text)),
        has_negation=detect_negation(text),
    )
