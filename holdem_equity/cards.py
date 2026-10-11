"""Card parsing and the deck.

A card is an int in 0..51: ``rank * 4 + suit``, where rank 0 = '2' ... 12 = 'A'
and suit 0..3 = 'c', 'd', 'h', 's'. Small ints index straight into numpy arrays,
which is what the vectorised evaluator and Monte Carlo code need.
"""

import re

import numpy as np

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK_SIZE = 52

_CARD_RE = re.compile(r"([2-9TJQKA])([cdhs])")


def parse_card(text: str) -> int:
    """'Ah' -> 50. Rank is case-insensitive, suit must be one of c/d/h/s."""
    text = text.strip()
    if len(text) != 2 or not _CARD_RE.fullmatch(text[0].upper() + text[1].lower()):
        raise ValueError(f"invalid card {text!r}: expected rank 2-9/T/J/Q/K/A + suit c/d/h/s, e.g. 'Ah'")
    return RANKS.index(text[0].upper()) * 4 + SUITS.index(text[1].lower())


def parse_cards(text: str) -> list[int]:
    """'AhKh' or 'Ah Kh' -> [50, 46]. Rejects duplicates."""
    compact = text.replace(" ", "")
    if len(compact) % 2:
        raise ValueError(f"invalid card string {text!r}: odd number of characters")
    cards = [parse_card(compact[i:i + 2]) for i in range(0, len(compact), 2)]
    if len(set(cards)) != len(cards):
        raise ValueError(f"duplicate card in {text!r}")
    return cards


def card_str(card: int) -> str:
    """50 -> 'Ah'."""
    if not 0 <= card < DECK_SIZE:
        raise ValueError(f"card index out of range: {card}")
    return RANKS[card // 4] + SUITS[card % 4]


def cards_str(cards) -> str:
    return "".join(card_str(int(c)) for c in cards)


def remaining_deck(*dealt) -> np.ndarray:
    """All cards not in any of the given card lists, as a sorted int array.

    Raises if the same card was dealt twice (e.g. both players hold Ah).
    """
    used = [int(c) for group in dealt for c in group]
    if len(set(used)) != len(used):
        raise ValueError("the same card appears more than once: " + cards_str(used))
    mask = np.ones(DECK_SIZE, dtype=bool)
    mask[used] = False
    return np.flatnonzero(mask)
