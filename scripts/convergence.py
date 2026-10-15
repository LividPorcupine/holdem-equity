"""Convergence study: how fast does Monte Carlo approach the exact answer?

For three preflop matchups, compare the mean |MC - exact| over many seeds with
the theory: for n independent deals, the error of a sample mean is roughly
normal with sd sigma/sqrt(n), so E|error| = sqrt(2/pi) * sigma / sqrt(n).

Run from the repo root:  python -m scripts.convergence
Writes figures/convergence.png.
"""

from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from holdem_equity.cards import parse_cards
from holdem_equity.equity import exact_equity, monte_carlo_equity

MATCHUPS = [("AhAs", "KdKc", "AA vs KK"), ("AhKh", "QsQd", "AKs vs QQ"), ("AhKd", "2s2c", "AKo vs 22")]
SAMPLE_SIZES = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
SEEDS = 30  # independent runs per (matchup, n); error is averaged over them
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # dataviz reference palette, slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
OUT = Path(__file__).resolve().parent.parent / "figures" / "convergence.png"


def _abs_error(task):
    hero, villain, exact, n, seed = task
    return abs(monte_carlo_equity(parse_cards(hero), parse_cards(villain), samples=n, seed=seed).equity - exact)


def _spread(values, min_ratio=1.3):
    """Nudge end-of-line label heights apart (on a log axis) so they don't overlap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] * min_ratio)
    return placed


def main():
    exact = {}
    for hero, villain, label in MATCHUPS:
        r = exact_equity(parse_cards(hero), parse_cards(villain))
        sigma = np.sqrt(r.win + r.tie / 4 - r.equity ** 2)  # sd of one deal's outcome x in {1, 0.5, 0}
        exact[label] = (r.equity, sigma)
        print(f"{label}: exact equity {r.equity:.4%}, per-deal sd {sigma:.4f}")

    tasks = [(h, v, exact[label][0], n, 10_000 * i + s)
             for i, (h, v, label) in enumerate(MATCHUPS) for n in SAMPLE_SIZES for s in range(SEEDS)]
    with Pool() as pool:
        errors = np.array(pool.map(_abs_error, tasks)).reshape(len(MATCHUPS), len(SAMPLE_SIZES), SEEDS)
    mean_error_pp = 100 * errors.mean(axis=2)

    n = np.array(SAMPLE_SIZES, dtype=float)
    label_y = _spread(mean_error_pp[:, -1])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for (_, _, label), color, observed, y in zip(MATCHUPS, COLORS, mean_error_pp, label_y):
        theory_pp = 100 * np.sqrt(2 / np.pi) * exact[label][1] / np.sqrt(n)
        ax.plot(n, theory_pp, color=color, lw=1, ls="--", alpha=0.8)
        ax.plot(n, observed, color=color, lw=2, marker="o", ms=5)
        ax.annotate(label, (n[-1] * 1.15, y), va="center", fontsize=9, color=color)  # two lines end together
        print(f"{label}: mean |error| at n={SAMPLE_SIZES[-1]:,} = {observed[-1]:.4f} pp "
              f"(theory {theory_pp[-1]:.4f} pp)")

    ax.set(xscale="log", yscale="log", xlabel="Monte Carlo samples (n)",
           ylabel="mean |MC − exact| (percentage points)")
    ax.set_title("100× more samples → 10× smaller error", loc="left", color=INK)
    # Matchups are labelled at the line ends, so the legend only explains the two line styles.
    ax.plot([], [], color=MUTED, lw=2, marker="o", ms=5, label=f"observed: mean |MC − exact| over {SEEDS} seeds")
    ax.plot([], [], color=MUTED, lw=1, ls="--", label="theory: √(2/π)·σ/√n, one per matchup")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_xlim(right=n[-1] * 3)  # room for the direct labels
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
