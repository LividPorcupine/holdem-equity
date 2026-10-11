"""Hand evaluation, built on treys.

This is the only module that knows about treys, so the evaluator can be swapped
out without touching the rest of the package.

Ranks follow treys: 1 (royal flush) .. 7462 (7-5-4-3-2 offsuit); lower is stronger.

Two entry points:
- ``evaluate(cards)``: one hand, via ``treys.Evaluator`` (simple, slow).
- ``evaluate_batch(hands)``: many hands at once, vectorised with numpy using
  treys' own lookup tables. Same answers (tests check this), ~8x faster.
"""

from itertools import combinations

import numpy as np
from treys import Card, Evaluator

from .cards import RANKS, SUITS, card_str

_evaluator = Evaluator()

# Our card index (0..51) -> treys' 32-bit card int.
TREYS_CARDS = np.array([Card.new(card_str(c)) for c in range(len(RANKS) * len(SUITS))], dtype=np.int32)


def _sorted_table(table: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    # int32 is enough: the largest product, 41^4 * 37 (four aces + a king), is ~1.05e8.
    keys = np.array(sorted(table), dtype=np.int32)
    values = np.array([table[k] for k in keys.tolist()], dtype=np.int16)
    return keys, values


# treys' Cactus-Kev tables map "product of the five rank primes" -> hand rank,
# one table for flushes and one for everything else. As sorted arrays they can
# be searched for a whole batch at once with np.searchsorted.
_FLUSH_KEYS, _FLUSH_RANKS = _sorted_table(_evaluator.table.flush_lookup)
_UNSUITED_KEYS, _UNSUITED_RANKS = _sorted_table(_evaluator.table.unsuited_lookup)
_SUBSETS = {n: np.array(list(combinations(range(n), 5))) for n in (5, 6, 7)}
_CHUNK = 50_000  # hands per numpy pass; keeps peak memory around 100 MB


def evaluate(cards) -> int:
    """Rank of the best 5-card hand among 5-7 cards (our 0..51 ints)."""
    if not 5 <= len(cards) <= 7:
        raise ValueError(f"need 5-7 cards, got {len(cards)}")
    treys_ints = [int(TREYS_CARDS[c]) for c in cards]
    return _evaluator.evaluate(treys_ints[:2], treys_ints[2:])


def _lookup(table_keys: np.ndarray, table_ranks: np.ndarray, products: np.ndarray) -> np.ndarray:
    return table_ranks[np.searchsorted(table_keys, products)]


def _evaluate_chunk(hands: np.ndarray) -> np.ndarray:
    cards = TREYS_CARDS[hands]                       # (n, k) treys ints
    five = cards[:, _SUBSETS[hands.shape[1]]]        # (n, subsets, 5)
    products = np.prod(five & 0xFF, axis=2, dtype=np.int32)       # low byte of a treys card = its rank prime
    is_flush = np.bitwise_and.reduce(five & 0xF000, axis=2) != 0  # bits 12-15 = suit; all five share one
    ranks = _lookup(_UNSUITED_KEYS, _UNSUITED_RANKS, products)
    # Flushes are rare (~0.2% of random 5-card hands), so look them up only where they occur.
    ranks[is_flush] = _lookup(_FLUSH_KEYS, _FLUSH_RANKS, products[is_flush])
    return ranks.min(axis=1)


def evaluate_batch(hands) -> np.ndarray:
    """Ranks for an (n, k) array of hands, k in 5..7. Returns an (n,) int16 array."""
    hands = np.asarray(hands)
    if hands.ndim != 2 or hands.shape[1] not in _SUBSETS:
        raise ValueError(f"expected shape (n, 5..7), got {hands.shape}")
    out = np.empty(len(hands), dtype=np.int16)
    for start in range(0, len(hands), _CHUNK):
        out[start:start + _CHUNK] = _evaluate_chunk(hands[start:start + _CHUNK])
    return out
