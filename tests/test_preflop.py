from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holdem_equity.preflop import (
    COMBOS, HAND_CLASSES, class_combos, class_vs_class, combo_pair_counts, equity_vs_random, grid_label,
)

DATA = Path(__file__).resolve().parent.parent / "data"


def test_169_classes_in_grid_order():
    assert len(HAND_CLASSES) == len(set(HAND_CLASSES)) == 169
    assert (grid_label(0, 0), grid_label(0, 1), grid_label(1, 0), grid_label(12, 12)) == ("AA", "AKs", "AKo", "22")


def test_combo_counts():
    assert len(class_combos("AA")) == 6
    assert len(class_combos("AKs")) == 4
    assert len(class_combos("AKo")) == 12
    every_hand = {tuple(sorted(c)) for combos in COMBOS.values() for c in combos.tolist()}
    assert len(every_hand) == 1326  # C(52, 2): every two-card hand exactly once


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        class_combos("KAs")


def test_combo_pair_counts():
    counts = combo_pair_counts()
    # Every hero combo leaves C(50, 2) = 1225 possible villain hands.
    assert np.array_equal(counts.sum(axis=1), [len(COMBOS[c]) * 1225 for c in HAND_CLASSES])
    aa = HAND_CLASSES.index("AA")
    assert counts[aa, aa] == 6  # each AA combo only avoids its complementary AA combo


def test_mirror_matchup_is_a_coin_flip():
    equity, se = class_vs_class("AKs", "AKs", 20_000, np.random.default_rng(0))
    assert abs(equity - 0.5) < 4 * se


def test_equity_vs_random_of_flat_table_is_half():
    flat = pd.DataFrame(0.5, index=HAND_CLASSES, columns=HAND_CLASSES)
    assert np.allclose(equity_vs_random(flat), 0.5)


# --- the shipped table (acceptance checks) --------------------------------

@pytest.fixture(scope="module")
def table():
    return pd.read_csv(DATA / "preflop_equity.csv", index_col="hero")


def test_table_shape_and_labels(table):
    assert table.shape == (169, 169)
    assert list(table.index) == list(table.columns) == HAND_CLASSES


def test_table_is_symmetric(table):
    eq = table.to_numpy()
    assert np.abs(eq + eq.T - 1).max() <= 0.01


def test_vs_random_extremes(table):
    vs_random = equity_vs_random(table)
    assert vs_random.idxmax() == "AA" and 0.845 < vs_random["AA"] < 0.86
    assert vs_random.idxmin() == "32o" and 0.315 < vs_random["32o"] < 0.33
    assert 0.34 < vs_random["72o"] < 0.355


def test_long_format_and_parquet_agree():
    long = pd.read_csv(DATA / "preflop_equity_long.csv")
    parquet = pd.read_parquet(DATA / "preflop_equity.parquet")
    assert len(long) == len(parquet) == 169 * 169
    assert np.allclose(long.equity, parquet.equity, atol=1e-6)
    assert parquet.loc[parquet.hero != parquet.villain, "std_error"].max() < 0.003  # < 0.3 pp per cell
