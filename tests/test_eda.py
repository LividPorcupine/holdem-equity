from scripts.eda import flush_split


def test_suited_premium_comes_from_flushes():
    """The README's explanation of why 32 gains more from being suited than AK.

    Suited hands make a flush about 3x as often and have over 90% equity with it.
    Each extra flush replaces a board where AK still has about 65% equity but 32
    only about 32%, so the flush is worth roughly twice as much to 32.
    """
    ak_s, ak_o = flush_split("AKs", 200_000, seed=0), flush_split("AKo", 200_000, seed=1)
    s32, o32 = flush_split("32s", 200_000, seed=2), flush_split("32o", 200_000, seed=3)
    for suited, offsuit in ((ak_s, ak_o), (s32, o32)):
        assert suited["p_flush"] > 2.5 * offsuit["p_flush"]
        assert suited["eq_flush"] > 0.9
    assert s32["eq_flush"] - s32["eq_no_flush"] > 0.5 > ak_s["eq_flush"] - ak_s["eq_no_flush"]
