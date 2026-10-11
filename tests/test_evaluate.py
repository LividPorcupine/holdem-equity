import numpy as np
import pytest

from holdem_equity.cards import parse_cards
from holdem_equity.evaluate import evaluate, evaluate_batch


def test_royal_flush_is_rank_1():
    assert evaluate(parse_cards("AhKhQhJhTh2c3d")) == 1


def test_worst_hand_is_rank_7462():
    assert evaluate(parse_cards("7c5d4h3s2c")) == 7462


def test_stronger_hand_has_lower_rank():
    board = parse_cards("Td9h2c3s4d")
    pair = evaluate(parse_cards("Th8c") + board)
    high_card = evaluate(parse_cards("Ah8c") + board)
    assert pair < high_card


@pytest.mark.parametrize("k", [5, 6, 7])
def test_batch_matches_treys(k):
    rng = np.random.default_rng(0)
    hands = rng.random((3000, 52)).argsort(axis=1)[:, :k]
    assert np.array_equal(evaluate_batch(hands), [evaluate(h) for h in hands])


def test_batch_rejects_bad_shape():
    with pytest.raises(ValueError):
        evaluate_batch(np.zeros((10, 4), dtype=int))
