"""Speed of the vectorised evaluator vs calling treys one hand at a time.

Run from the repo root:  python -m scripts.benchmark [--hands N] [--seed S]
Both evaluators rank the same random 7-card hands; the script checks they agree.
"""

import argparse
import time

import numpy as np

from holdem_equity.evaluate import evaluate, evaluate_batch


def best_of(repeats, fn):
    """Fastest of `repeats` runs, in seconds (the least disturbed by other processes)."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return min(times), result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hands", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    hands = np.argsort(rng.random((args.hands, 52)), axis=1)[:, :7].astype(np.int8)  # 7 distinct cards per row

    loop_s, loop_ranks = best_of(3, lambda: np.array([evaluate(h) for h in hands.tolist()]))
    batch_s, batch_ranks = best_of(3, lambda: evaluate_batch(hands))
    assert np.array_equal(loop_ranks, batch_ranks), "evaluators disagree"

    print(f"{args.hands:,} random 7-card hands, identical ranks from both evaluators (best of 3 runs)")
    print(f"  treys, one hand at a time: {args.hands / loop_s / 1e6:.2f}M hands/s")
    print(f"  evaluate_batch (numpy):    {args.hands / batch_s / 1e6:.2f}M hands/s")
    print(f"  speed-up: {loop_s / batch_s:.1f}x")


if __name__ == "__main__":
    main()
