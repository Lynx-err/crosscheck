"""Deterministic random-score baseline."""

from __future__ import annotations

import hashlib
import random


def random_score(seed: int, item_id: str) -> float:
    """Return a reproducible pseudo-random hallucination score for an item."""

    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random()
