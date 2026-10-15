from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from holdem_equity.equity import monte_carlo_equity
from holdem_equity.preflop import (
    COMBOS, HAND_CLASSES, class_combos, class_vs_class, combo_pair_counts, equity_vs_random, exact_class_vs_class,
    grid_label,
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


def test_class_vs_class_matches_exact():
    exact = exact_class_vs_class("AKs", "QQ")  # 2 suit patterns, ~6 s
    assert exact == pytest.approx(0.460485, abs=1e-6)
    equity, se = class_vs_class("AKs", "QQ", 40_000, np.random.default_rng(0))
    assert abs(equity - exact) < 4 * se


def test_equity_vs_random_of_flat_table_is_half():
    flat = pd.DataFrame(0.5, index=HAND_CLASSES, columns=HAND_CLASSES)
    vs_random = equity_vs_random(flat, flat * 0)
    assert np.allclose(vs_random.equity, 0.5) and (vs_random.std_error == 0).all()


# --- the shipped table (acceptance checks) --------------------------------

@pytest.fixture(scope="module")
def table():
    return pd.read_csv(DATA / "preflop_equity.csv", index_col="hero")


@pytest.fixture(scope="module")
def vs_random(table):
    long = pd.read_parquet(DATA / "preflop_equity.parquet")
    se = long.pivot(index="hero", columns="villain", values="std_error").loc[HAND_CLASSES, HAND_CLASSES]
    return equity_vs_random(table, se)


def test_table_shape_and_labels(table):
    assert table.shape == (169, 169)
    assert list(table.index) == list(table.columns) == HAND_CLASSES


def test_table_is_symmetric(table):
    eq = table.to_numpy()
    assert np.abs(eq + eq.T - 1).max() <= 0.01


def test_vs_random_extremes(vs_random):
    eq = vs_random.equity
    assert eq.idxmax() == "AA" and 0.845 < eq["AA"] < 0.86
    assert eq.idxmin() == "32o" and 0.315 < eq["32o"] < 0.33
    assert 0.34 < eq["72o"] < 0.355


@pytest.mark.parametrize("hand", ["32o", "72o", "AA", "22"])
def test_vs_random_matches_direct_simulation(vs_random, hand):
    # Independent of the table: simulate one combo (all combos of a class are
    # equivalent preflop) against a random hand, and compare as a z-score.
    direct = monte_carlo_equity(COMBOS[hand][0].tolist(), None, samples=1_000_000, seed=11)
    z = (direct.equity - vs_random.equity[hand]) / np.hypot(direct.std_error, vs_random.std_error[hand])
    assert abs(z) < 4


def test_long_format_and_parquet_agree():
    long = pd.read_csv(DATA / "preflop_equity_long.csv")
    parquet = pd.read_parquet(DATA / "preflop_equity.parquet")
    assert len(long) == len(parquet) == 169 * 169
    assert np.allclose(long.equity, parquet.equity, atol=1e-6)
    assert parquet.loc[parquet.hero != parquet.villain, "std_error"].max() < 0.003  # < 0.3 pp per cell
