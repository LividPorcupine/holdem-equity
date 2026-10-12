"""The 169 preflop hand classes and the 169x169 class-vs-class equity table.

Preflop, suits only matter through "same suit or not", so the 1,326 two-card
hands collapse into 169 classes:
  13 pairs   (e.g. 'AA')  x  6 combos  =  78
  78 suited  (e.g. 'AKs') x  4 combos  = 312
  78 offsuit (e.g. 'AKo') x 12 combos  = 936
                                       = 1,326

Classes are listed in the standard 13x13 grid order (row-major, A first):
the diagonal holds pairs, above it suited hands, below it offsuit hands.
"""

from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

from .cards import RANKS
from .equity import BOARD_SIZE, showdown_outcomes

RANKS_DESC = RANKS[::-1]  # 'AKQJT98765432'
GRID = len(RANKS_DESC)


def grid_label(row: int, col: int) -> str:
    """Class at (row, col) of the 13x13 grid: (0, 0) = 'AA', (0, 1) = 'AKs', (1, 0) = 'AKo'."""
    if row == col:
        return RANKS_DESC[row] * 2
    hi, lo = sorted((row, col))  # smaller index = higher rank
    return RANKS_DESC[hi] + RANKS_DESC[lo] + ("s" if row < col else "o")


HAND_CLASSES = [grid_label(r, c) for r in range(GRID) for c in range(GRID)]


def class_combos(label: str) -> np.ndarray:
    """Every concrete two-card hand in a class, as a (k, 2) array of card ints."""
    if label not in HAND_CLASSES:
        raise ValueError(f"unknown hand class {label!r}; expected e.g. 'AA', 'AKs', 'T9o'")
    hi, lo = RANKS.index(label[0]), RANKS.index(label[1])
    if hi == lo:
        pairs = [(a, b) for a, b in combinations(range(4), 2)]  # two different suits
    elif label[2] == "s":
        pairs = [(s, s) for s in range(4)]
    else:
        pairs = [(a, b) for a in range(4) for b in range(4) if a != b]
    return np.array([(hi * 4 + a, lo * 4 + b) for a, b in pairs], dtype=np.int8)


COMBOS = {label: class_combos(label) for label in HAND_CLASSES}


def _masks(combos: np.ndarray) -> np.ndarray:
    """52-bit card-set mask per combo; two hands conflict iff their masks share a bit."""
    return (np.uint64(1) << combos[:, 0].astype(np.uint64)) | (np.uint64(1) << combos[:, 1].astype(np.uint64))


def _valid_pairs(hero: str, villain: str) -> np.ndarray:
    """Index pairs (a, b) such that combo a of hero and combo b of villain share no card."""
    clash = (_masks(COMBOS[hero])[:, None] & _masks(COMBOS[villain])[None, :]) != 0
    return np.argwhere(~clash)


def class_vs_class(hero: str, villain: str, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    """Monte Carlo equity of class `hero` vs class `villain`, with its standard error.

    Each sample draws a non-conflicting (hero combo, villain combo) pair uniformly,
    then 5 board cards uniformly from the remaining 48.
    """
    pairs = _valid_pairs(hero, villain)
    pick = pairs[rng.integers(len(pairs), size=samples)]
    hero_cards, villain_cards = COMBOS[hero][pick[:, 0]], COMBOS[villain][pick[:, 1]]

    # Random sort keys per card; dealt cards get a key above every random one, so
    # the 5 smallest keys are a uniform 5-card board from the other 48 cards.
    keys = rng.random((samples, 52))
    keys[np.arange(samples)[:, None], np.hstack([hero_cards, villain_cards])] = 2.0
    boards = np.argpartition(keys, BOARD_SIZE, axis=1)[:, :BOARD_SIZE].astype(np.int8)

    x = showdown_outcomes(hero_cards, villain_cards, boards)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(samples))


def _row(task):
    """Cells (i, j) for j > i: one unit of parallel work."""
    i, samples, seed_seq = task
    rng = np.random.default_rng(seed_seq)
    return i, [class_vs_class(HAND_CLASSES[i], HAND_CLASSES[j], samples, rng) for j in range(i + 1, len(HAND_CLASSES))]


def build_table(samples: int = 40_000, seed: int = 2026, processes: int | None = None):
    """Equity and standard-error tables (169x169 DataFrames, rows = hero class).

    Only cells above the diagonal are simulated; eq[j][i] = 1 - eq[i][j] because
    heads-up the two shares of the pot always sum to 1 (same samples, same SE).
    Diagonal cells (e.g. AA vs AA) are exactly 0.5 by symmetry.
    Each row gets its own spawned seed, so results don't depend on `processes`.
    """
    n = len(HAND_CLASSES)
    eq, se = np.full((n, n), 0.5), np.zeros((n, n))
    seeds = np.random.SeedSequence(seed).spawn(n)
    tasks = [(i, samples, seeds[i]) for i in range(n - 1)]
    with Pool(processes) as pool:
        for i, cells in pool.imap_unordered(_row, tasks):
            for j, (e, s) in enumerate(cells, start=i + 1):
                eq[i, j], eq[j, i] = e, 1 - e
                se[i, j] = se[j, i] = s
    frame = lambda a: pd.DataFrame(a, index=pd.Index(HAND_CLASSES, name="hero"), columns=HAND_CLASSES)
    return frame(eq), frame(se)


def combo_pair_counts() -> np.ndarray:
    """counts[i, j] = number of non-conflicting (combo of class i, combo of class j) pairs."""
    masks = [_masks(COMBOS[c]) for c in HAND_CLASSES]
    return np.array([[np.count_nonzero((a[:, None] & b[None, :]) == 0) for b in masks] for a in masks])


def equity_vs_random(table: pd.DataFrame) -> pd.Series:
    """Each class's equity against a uniformly random opponent hand, from the table.

    Dealing hero a random combo of class i and villain a random hand from the
    other 50 cards makes every non-conflicting (hero combo, villain combo) pair
    equally likely, so class j appears with weight counts[i, j]:
        eq_vs_random(i) = sum_j counts[i, j] * eq[i, j] / sum_j counts[i, j]
    """
    counts = combo_pair_counts()
    values = (counts * table.to_numpy()).sum(axis=1) / counts.sum(axis=1)
    return pd.Series(values, index=table.index, name="equity_vs_random")


def to_long(eq: pd.DataFrame, se: pd.DataFrame, samples: int) -> pd.DataFrame:
    """One row per (hero, villain) cell: hero, villain, equity, std_error, samples."""
    long = eq.stack().rename("equity").to_frame()
    long["std_error"] = se.stack()
    long["samples"] = np.where(long.index.get_level_values(0) == long.index.get_level_values(1), 0, samples)
    return long.rename_axis(["hero", "villain"]).reset_index()
