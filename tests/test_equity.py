import pytest

from holdem_equity.cards import parse_cards as cards
from holdem_equity.equity import exact_equity

# (hero, villain, spec's sanity band for hero equity)
KNOWN_MATCHUPS = [
    ("AhAs", "KdKc", (0.81, 0.83)),  # AA vs KK
    ("AhKh", "QsQd", (0.45, 0.47)),  # AKs vs QQ
    ("AhKd", "2s2c", (0.46, 0.48)),  # AKo vs 22
]


@pytest.mark.parametrize("hero, villain, band", KNOWN_MATCHUPS)
def test_exact_preflop_known_values(hero, villain, band):
    result = exact_equity(cards(hero), cards(villain))
    assert band[0] <= result.equity <= band[1]
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
