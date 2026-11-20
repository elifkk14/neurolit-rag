"""Split an abstract into chunks that fit the embedding model.

bge-small reads at most 512 tokens; longer input is silently cut off. ~300 English words is
roughly 400 tokens, which leaves room for the title. We split on sentence boundaries (never
in the middle of a sentence) and repeat the last sentence of a chunk at the start of the
next one, so a fact that spans two sentences is not lost at the border.
"""

from __future__ import annotations

import re

MAX_WORDS = 300

# End of sentence: . ! or ? followed by whitespace and an upper-case letter or digit.
# Avoids splitting "e.g. the" or "p < 0.05 in", which are common in abstracts.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.replace("\n", " ")) if s.strip()]


def chunk_abstract(title: str, abstract: str, max_words: int = MAX_WORDS) -> list[str]:
    """Return chunk texts. Each chunk starts with the title, so it makes sense on its own."""
    budget = max_words - len(title.split())
    sentences = split_sentences(abstract)

    chunks: list[list[str]] = []
    current: list[str] = []
    words = 0
    for sentence in sentences:
        n = len(sentence.split())
        if current and words + n > budget:
            chunks.append(current)
            current = [current[-1]]  # one-sentence overlap
            words = len(current[0].split())
        current.append(sentence)
        words += n
    if current:
        chunks.append(current)

    return [f"{title}\n{' '.join(c)}" for c in chunks]
