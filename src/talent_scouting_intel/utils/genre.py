from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_sim(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a.keys()) & set(b.keys())
    dot = sum(a[token] * b[token] for token in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def classify_genre(text: str, prototypes: dict[str, list[str]]) -> tuple[str, float]:
    doc_vec = Counter(tokenize(text))
    if not doc_vec:
        return "unknown", 0.0

    best_genre = "unknown"
    best_score = 0.0
    for genre, words in prototypes.items():
        proto_vec = Counter(tokenize(" ".join(words)))
        score = cosine_sim(doc_vec, proto_vec)
        if score > best_score:
            best_score = score
            best_genre = genre
    return best_genre, best_score
