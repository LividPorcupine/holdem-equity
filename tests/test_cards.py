import pytest

from holdem_equity.cards import card_str, parse_card, parse_cards, remaining_deck


def test_parse_card_round_trip():
    for text in ["2c", "Td", "Ah", "Ks", "9h"]:
        assert card_str(parse_card(text)) == text


def test_parse_card_index_layout():
    assert parse_card("2c") == 0
    assert parse_card("As") == 51


def test_parse_card_is_case_insensitive():
    assert parse_card("ah") == parse_card("AH") == parse_card("Ah")


@pytest.mark.parametrize("bad", ["", "A", "Ax", "1h", "10h", "Ahh", "Zz"])
def test_parse_card_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_card(bad)


def test_parse_cards():
    assert parse_cards("AhKh") == [parse_card("Ah"), parse_card("Kh")]
    assert parse_cards("Td 9h 2c") == parse_cards("Td9h2c")


@pytest.mark.parametrize("bad", ["AhAh", "AhK", "AhKx"])
def test_parse_cards_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        parse_cards(bad)


def test_remaining_deck_removes_dealt_cards():
    hero, villain = parse_cards("AhKh"), parse_cards("QsQd")
    deck = remaining_deck(hero, villain)
    assert len(deck) == 48
    assert not set(hero + villain) & set(deck.tolist())


def test_remaining_deck_rejects_conflicts():
    with pytest.raises(ValueError):
        remaining_deck(parse_cards("AhKh"), parse_cards("AhQd"))
