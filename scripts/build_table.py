"""Build the 169x169 preflop equity table and save it to data/.

Run from the repo root:  python -m scripts.build_table [--samples N] [--seed S]

Outputs (see data/README.md for the format):
  data/preflop_equity.csv          169x169 matrix, rows = hero class, columns = villain class
  data/preflop_equity_long.csv     one row per cell: hero, villain, equity, std_error, samples
  data/preflop_equity.parquet      same as the long CSV
"""

import argparse
import time
from pathlib import Path

from holdem_equity.preflop import build_table, equity_vs_random, to_long

DATA = Path(__file__).resolve().parent.parent / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=40_000, help="Monte Carlo samples per cell")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    start = time.perf_counter()
    eq, se = build_table(samples=args.samples, seed=args.seed)
    elapsed = time.perf_counter() - start

    DATA.mkdir(exist_ok=True)
    eq.round(6).to_csv(DATA / "preflop_equity.csv")
    long = to_long(eq, se, args.samples)
    long.round(6).to_csv(DATA / "preflop_equity_long.csv", index=False)
    long.to_parquet(DATA / "preflop_equity.parquet", index=False)

    vs_random = equity_vs_random(eq, se).equity.sort_values(ascending=False)
    off_diagonal = long[long.hero != long.villain]
    print(f"built in {elapsed:.0f} s with {args.samples:,} samples per cell, seed {args.seed}")
    print(f"max std error {100 * off_diagonal.std_error.max():.3f} pp, "
          f"mean {100 * off_diagonal.std_error.mean():.3f} pp")
    print(f"max |eq[i][j] + eq[j][i] - 1| = {abs(eq.to_numpy() + eq.to_numpy().T - 1).max():.2e}")
    print("vs random, top 3:", ", ".join(f"{h} {v:.2%}" for h, v in vs_random.head(3).items()))
    print("vs random, bottom 3:", ", ".join(f"{h} {v:.2%}" for h, v in vs_random.tail(3).items()))
    print(f"72o vs random: {vs_random['72o']:.2%}")


if __name__ == "__main__":
    main()
