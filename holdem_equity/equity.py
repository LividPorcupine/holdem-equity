"""Heads-up all-in equity: exact enumeration and Monte Carlo.

Equity of a hand = P(win) + P(tie) / 2, i.e. its expected share of the pot.
The villain is either a known two-card hand or ``None`` for "a random hand".
Cards are our 0..51 ints (see cards.py).
"""

from dataclasses import dataclass
from itertools import chain, combinations
from math import comb

import numpy as np

from .cards import remaining_deck
from .evaluate import evaluate_batch

BOARD_SIZE = 5
MAX_EXACT_SHOWDOWNS = 20_000_000  # ~20 s; preflop vs a random hand would be ~2.1e9


@dataclass(frozen=True)
class EquityResult:
    equity: float     # hero's share of the pot: win + tie / 2
    win: float        # P(hero wins)
    tie: float        # P(split pot)
    std_error: float  # standard error of `equity`; 0 for exact results
    showdowns: int    # boards (x villain hands) enumerated or sampled


def _check_inputs(hero, villain, board):
    if len(hero) != 2 or (villain is not None and len(villain) != 2):
        raise ValueError("each hand must be exactly 2 cards")
    if len(board) > BOARD_SIZE:
        raise ValueError(f"board has at most {BOARD_SIZE} cards, got {len(board)}")


def _all_combos(cards: np.ndarray, k: int) -> np.ndarray:
    """Every k-card combination of `cards` as an (n, k) int8 array."""
    n = comb(len(cards), k)
    flat = np.fromiter(chain.from_iterable(combinations(cards.tolist(), k)), dtype=np.int8, count=n * k)
    return flat.reshape(n, k)


def showdown_outcomes(hero, villain_cards: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Hero's result per showdown: 1 win, 0.5 tie, 0 loss.

    hero: (2,) for a fixed hand or (n, 2); villain_cards: (n, 2); boards: (n, 5).
    Row i of the arrays is one complete deal.
    """
    n = len(boards)
    hero_rank = evaluate_batch(np.hstack([np.broadcast_to(np.asarray(hero, dtype=np.int8), (n, 2)), boards]))
    villain_rank = evaluate_batch(np.hstack([villain_cards, boards]))
    return np.where(hero_rank < villain_rank, 1.0, np.where(hero_rank == villain_rank, 0.5, 0.0))


def _tally(outcomes: np.ndarray, totals: np.ndarray) -> np.ndarray:
    """Add [wins, ties, showdowns] from one batch to the running totals."""
    return totals + [np.count_nonzero(outcomes == 1.0), np.count_nonzero(outcomes == 0.5), len(outcomes)]


def exact_equity(hero, villain=None, board=()) -> EquityResult:
    """Enumerate every remaining board (and every villain hand if villain is None)."""
    _check_inputs(hero, villain, board)
    board = list(board)
    missing = BOARD_SIZE - len(board)
    deck = remaining_deck(hero, villain or [], board)

    if villain is None:
        showdowns = comb(len(deck), 2) * comb(len(deck) - 2, missing)
    else:
        showdowns = comb(len(deck), missing)
    if showdowns > MAX_EXACT_SHOWDOWNS:
        raise ValueError(
            f"exact enumeration needs {showdowns:,} showdowns (limit {MAX_EXACT_SHOWDOWNS:,}); "
            "use monte_carlo_equity instead"
        )

    villains = [villain] if villain is not None else combinations(deck.tolist(), 2)
    totals = np.zeros(3, dtype=np.int64)
    for v in villains:
        runouts = _all_combos(remaining_deck(hero, v, board), missing)
        boards = np.hstack([np.broadcast_to(np.array(board, dtype=np.int8), (len(runouts), len(board))), runouts])
        totals = _tally(showdown_outcomes(hero, np.broadcast_to(np.array(v, dtype=np.int8), (len(boards), 2)), boards), totals)

    wins, ties, n = totals.tolist()
    return EquityResult(equity=(wins + ties / 2) / n, win=wins / n, tie=ties / n, std_error=0.0, showdowns=n)


_MC_CHUNK = 100_000  # deals per numpy pass


def monte_carlo_equity(hero, villain=None, board=(), samples=100_000, seed=None) -> EquityResult:
    """Estimate equity from `samples` random deals of the unseen cards.

    Each deal's outcome x is 1 / 0.5 / 0; equity = mean(x) and
    std_error = std(x) / sqrt(samples), which shrinks like 1 / sqrt(samples).
    The same seed always gives the same result.
    """
    _check_inputs(hero, villain, board)
    if samples < 2:
        raise ValueError("need at least 2 samples to estimate a standard error")
    rng = np.random.default_rng(seed)
    board = np.array(board, dtype=np.int8)
    deck = remaining_deck(hero, villain or [], board.tolist()).astype(np.int8)
    deal_size = (BOARD_SIZE - len(board)) + (2 if villain is None else 0)

    outcomes = np.empty(samples)
    for start in range(0, samples, _MC_CHUNK):
        n = min(_MC_CHUNK, samples - start)
        # Shuffle the unseen cards independently for each deal and take the top few.
        deals = rng.permuted(np.tile(deck, (n, 1)), axis=1)[:, :deal_size]
        if villain is None:
            villain_cards, runouts = deals[:, :2], deals[:, 2:]
        else:
            villain_cards, runouts = np.broadcast_to(np.array(villain, dtype=np.int8), (n, 2)), deals
        boards = np.hstack([np.broadcast_to(board, (n, len(board))), runouts])
        outcomes[start:start + n] = showdown_outcomes(hero, villain_cards, boards)

    return EquityResult(
        equity=float(outcomes.mean()),
        win=float(np.mean(outcomes == 1.0)),
        tie=float(np.mean(outcomes == 0.5)),
        std_error=float(outcomes.std(ddof=1) / np.sqrt(samples)),
        showdowns=samples,
    )
