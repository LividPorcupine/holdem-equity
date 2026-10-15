"""Speed of the vectorised evaluator vs calling treys one hand at a time.

Run from the repo root:  python -m scripts.benchmark
Both evaluators rank the same random 7-card hands; the script checks they agree.
"""

import time

import numpy as np

from holdem_equity.evaluate import evaluate, evaluate_batch

HANDS = 200_000


def best_of_3(fn):
    """Fastest of 3 runs in seconds (the one least disturbed by other processes), and the result."""
    times = []
    for _ in range(3):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return min(times), result


def main():
    rng = np.random.default_rng(0)
    hands = np.argsort(rng.random((HANDS, 52)), axis=1)[:, :7].astype(np.int8)  # 7 distinct cards per row

    loop_s, loop_ranks = best_of_3(lambda: np.array([evaluate(h) for h in hands.tolist()]))
    batch_s, batch_ranks = best_of_3(lambda: evaluate_batch(hands))
    assert np.array_equal(loop_ranks, batch_ranks), "evaluators disagree"

    print(f"{HANDS:,} random 7-card hands, identical ranks from both evaluators (best of 3 runs)")
    print(f"  treys, one hand at a time: {HANDS / loop_s / 1e6:.2f}M hands/s")
    print(f"  evaluate_batch (numpy):    {HANDS / batch_s / 1e6:.2f}M hands/s")
    print(f"  speed-up: {loop_s / batch_s:.1f}x")


if __name__ == "__main__":
    main()
