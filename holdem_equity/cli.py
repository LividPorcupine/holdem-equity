"""Command line: python -m holdem_equity AhKh QsQd [--board Td9h2c] [--exact | --samples N] [--seed S]"""

import argparse

from .cards import cards_str, parse_cards
from .equity import exact_equity, monte_carlo_equity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m holdem_equity",
        description="Heads-up all-in equity of two hands. Use 'random' as the second hand for a random opponent.",
    )
    parser.add_argument("hero", help="two cards, e.g. AhKh")
    parser.add_argument("villain", help="two cards, e.g. QsQd, or 'random'")
    parser.add_argument("--board", default="", help="0, 3, 4 or 5 board cards, e.g. Td9h2c")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--exact", action="store_true", help="enumerate every runout instead of sampling")
    mode.add_argument("--samples", type=int, default=100_000, help="Monte Carlo samples (default 100,000)")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible Monte Carlo")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        hero = parse_cards(args.hero)
        villain = None if args.villain.lower() == "random" else parse_cards(args.villain)
        board = parse_cards(args.board) if args.board else []
        if len(board) not in (0, 3, 4, 5):
            raise ValueError(f"board must have 0, 3, 4 or 5 cards, got {len(board)}")
        if args.exact:
            result = exact_equity(hero, villain, board)
            method = f"exact, {result.showdowns:,} showdowns"
        else:
            result = monte_carlo_equity(hero, villain, board, samples=args.samples, seed=args.seed)
            method = f"Monte Carlo, {result.showdowns:,} samples" + (f", seed {args.seed}" if args.seed is not None else "")
    except ValueError as err:
        parser.error(str(err))

    villain_name = "random" if villain is None else cards_str(villain)
    margin = f" +/- {100 * result.std_error:.2f}" if result.std_error else ""  # ASCII: Windows consoles mangle "±"
    print(f"{cards_str(hero)} vs {villain_name}, board: {cards_str(board) or '(none)'}  [{method}]")
    print(f"  {cards_str(hero):<7}{result.equity:7.2%}{margin}   (win {result.win:.2%}, tie {result.tie:.2%})")
    print(f"  {villain_name:<7}{1 - result.equity:7.2%}{margin}")
