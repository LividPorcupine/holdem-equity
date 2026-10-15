from functools import cache

import numpy as np
import pytest

from holdem_equity.cards import parse_cards as cards
from holdem_equity.equity import exact_equity, monte_carlo_equity

# (hero, villain, spec's sanity band for hero equity, exact value over all C(48, 5) boards)
KNOWN_MATCHUPS = [
    ("AhAs", "KdKc", (0.81, 0.83), 0.8125548967940273),  # AA vs KK
    ("AhKh", "QsQd", (0.45, 0.47), 0.46214457245909607),  # AKs vs QQ
    ("AhKd", "2s2c", (0.46, 0.48), 0.4695965202440688),  # AKo vs 22
]


@cache
def exact_preflop(hero, villain):
    """Each exact preflop matchup takes ~3 s; compute it once per test run."""
    return exact_equity(cards(hero), cards(villain))


@pytest.mark.parametrize("hero, villain, band, value", KNOWN_MATCHUPS)
def test_exact_preflop_known_values(hero, villain, band, value):
    result = exact_preflop(hero, villain)
    assert band[0] <= result.equity <= band[1]
    assert result.equity == pytest.approx(value, abs=1e-12)
    assert result.showdowns == 1_712_304  # C(48, 5) boards
    assert result.equity == pytest.approx(result.win + result.tie / 2)


def test_exact_river_win_and_split():
    board = cards("QhJhTh2c3d")
    assert exact_equity(cards("AhKh"), cards("AsAc"), board).equity == 1.0  # royal flush
    assert exact_equity(cards("4c5c"), cards("4d5d"), board).equity == 0.5  # identical hands


def test_exact_turn_counts_every_river():
    result = exact_equity(cards("AhKh"), cards("QsQd"), cards("Td9h2c3s"))
    assert result.showdowns == 44


def test_exact_vs_random_on_flop():
    result = exact_equity(cards("AhKh"), None, cards("Td9h2c"))
    assert result.showdowns == 1081 * 990  # C(47, 2) villain hands x C(45, 2) runouts
    assert 0 < result.equity < 1


def test_exact_refuses_huge_enumeration():
    with pytest.raises(ValueError, match="monte_carlo_equity"):
        exact_equity(cards("AhKh"), None)


def test_rejects_card_conflicts():
    with pytest.raises(ValueError):
        exact_equity(cards("AhKh"), cards("AhQd"))


# --- Monte Carlo ---------------------------------------------------------

@pytest.mark.parametrize("hero, villain, band, value", KNOWN_MATCHUPS)
def test_monte_carlo_within_4_standard_errors_of_exact(hero, villain, band, value):
    exact = exact_preflop(hero, villain).equity
    estimate = monte_carlo_equity(cards(hero), cards(villain), samples=100_000, seed=1)
    assert abs(estimate.equity - exact) < 4 * estimate.std_error


def test_monte_carlo_is_reproducible_with_a_seed():
    a = monte_carlo_equity(cards("AhKh"), None, samples=5_000, seed=42)
    b = monte_carlo_equity(cards("AhKh"), None, samples=5_000, seed=42)
    assert a == b


def test_monte_carlo_standard_error_shrinks_like_root_n():
    small = monte_carlo_equity(cards("AhKh"), cards("QsQd"), samples=10_000, seed=0)
    large = monte_carlo_equity(cards("AhKh"), cards("QsQd"), samples=160_000, seed=0)
    assert large.std_error == pytest.approx(small.std_error / 4, rel=0.05)  # 16x samples -> 1/4 error


def test_monte_carlo_vs_random_agrees_with_exact():
    board = cards("Td9h2c")
    exact = exact_equity(cards("AhKh"), None, board).equity
    estimate = monte_carlo_equity(cards("AhKh"), None, board, samples=100_000, seed=7)
    assert abs(estimate.equity - exact) < 4 * estimate.std_error


def test_accepts_numpy_array_hands():
    hero, villain, board = np.array(cards("AhKh")), np.array(cards("QsQd")), np.array(cards("Td9h2c3s"))
    assert exact_equity(hero, villain, board) == exact_equity(cards("AhKh"), cards("QsQd"), cards("Td9h2c3s"))
    assert monte_carlo_equity(hero, villain, samples=1_000, seed=3) == \
        monte_carlo_equity(cards("AhKh"), cards("QsQd"), samples=1_000, seed=3)
