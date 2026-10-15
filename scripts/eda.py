"""Exploratory analysis of the preflop table.

Run from the repo root (after build_table):  python -m scripts.eda
Prints the findings as markdown, writes data/equity_vs_random.csv and figures/heatmap.png.
Derived numbers carry standard errors propagated from the table's per-cell SEs.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from holdem_equity.cards import remaining_deck
from holdem_equity.equity import showdown_outcomes
from holdem_equity.evaluate import evaluate_batch
from holdem_equity.preflop import COMBOS, GRID, HAND_CLASSES, RANKS_DESC, combo_pair_counts, equity_vs_random

ROOT = Path(__file__).resolve().parent.parent
# dataviz reference palette: sequential blue ramp, steps 100 -> 700
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#0b0b0b", "#52514e"
# treys ranks of hands that are flushes: straight flushes 1-10, plain flushes 323-1599.
FLUSH_RANKS = ((1, 10), (323, 1599))


def hand_type(label: str) -> str:
    return "pair" if len(label) == 2 else {"s": "suited", "o": "offsuit"}[label[2]]


def load_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 169x169 equity table and its per-cell standard errors."""
    table = pd.read_csv(ROOT / "data" / "preflop_equity.csv", index_col="hero")
    long = pd.read_parquet(ROOT / "data" / "preflop_equity.parquet")
    se = long.pivot(index="hero", columns="villain", values="std_error").loc[HAND_CLASSES, HAND_CLASSES]
    return table, se


def difference_se(vs_random: pd.DataFrame, se: pd.DataFrame, a: str, b: str) -> float:
    """Standard error of vs_random[a] - vs_random[b].

    The two rows share one simulated cell: eq[a][b] = 1 - eq[b][a]. It enters the
    difference with the same sign from both rows, so its variance adds 2*w_a*w_b*se^2
    on top of the two rows' own variances.
    """
    counts = combo_pair_counts()
    i, j = HAND_CLASSES.index(a), HAND_CLASSES.index(b)
    w_a, w_b = counts[i, j] / counts[i].sum(), counts[j, i] / counts[j].sum()
    shared = 2 * w_a * w_b * se.iloc[i, j] ** 2
    return float(np.sqrt(vs_random.std_error[a] ** 2 + vs_random.std_error[b] ** 2 + shared))


def flush_split(label: str, samples: int, seed: int) -> dict[str, float]:
    """Direct simulation of class `label` vs a random hand, split by whether hero ends with a flush.

    Returns P(flush) and hero's equity on the boards with and without one.
    """
    hero = COMBOS[label][0]  # preflop, every combo of a class is equivalent
    rng = np.random.default_rng(seed)
    deck = remaining_deck(hero.tolist(), [], []).astype(np.int8)
    deals = rng.permuted(np.tile(deck, (samples, 1)), axis=1)[:, :7]
    villain, boards = deals[:, :2], deals[:, 2:]
    x = showdown_outcomes(hero, villain, boards)
    rank = evaluate_batch(np.hstack([np.broadcast_to(hero, (samples, 2)), boards]))
    flush = np.zeros(samples, dtype=bool)
    for lo, hi in FLUSH_RANKS:
        flush |= (rank >= lo) & (rank <= hi)
    return {"p_flush": flush.mean(), "eq_flush": x[flush].mean(), "eq_no_flush": x[~flush].mean()}


def main():
    table, se = load_table()
    df = equity_vs_random(table, se)
    df["type"] = [hand_type(h) for h in df.index]
    df["combos"] = [len(COMBOS[h]) for h in df.index]
    df = df.sort_values("equity", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.round(6).to_csv(ROOT / "data" / "equity_vs_random.csv", index_label="hand")

    print("## Findings\n")
    print("**Strongest / weakest vs a random hand** (± 1 SE)\n")
    print("| Rank | Hand | Equity |  | Rank | Hand | Equity |\n|---|---|---|---|---|---|---|")
    for (h1, r1), (h2, r2) in zip(df.head(5).iterrows(), df.tail(5).iterrows()):
        print(f"| {r1['rank']} | {h1} | {r1.equity:.2%} ± {100 * r1.std_error:.2f} "
              f"|  | {r2['rank']} | {h2} | {r2.equity:.2%} ± {100 * r2.std_error:.2f} |")
    diff, diff_se = df.equity["72o"] - df.equity["32o"], difference_se(df, se, "72o", "32o")
    print(f"\n72o - 32o = {100 * diff:.2f} ± {100 * diff_se:.2f} pp (z = {diff / diff_se:.0f}).")

    weighted = np.average(df.equity, weights=df.combos)
    print(f"\nCombo-weighted average over all 1,326 hands: {weighted:.4f}. This holds by construction "
          "(eq[j][i] = 1 - eq[i][j]), so it is not a test of the table.\n")

    print("**Average equity by type** (mean over classes)\n")
    print("| Type | Classes | Mean | Min | Max |\n|---|---|---|---|---|")
    for t, g in df.groupby("type", sort=False):
        print(f"| {t} | {len(g)} | {g.equity.mean():.1%} | {g.equity.min():.1%} ({g.equity.idxmin()}) "
              f"| {g.equity.max():.1%} ({g.equity.idxmax()}) |")

    eq = df.equity
    pairs = [hi + lo for i, hi in enumerate(RANKS_DESC) for lo in RANKS_DESC[i + 1:]]
    premium = pd.Series({p: eq[p + "s"] - eq[p + "o"] for p in pairs})
    premium_se = pd.Series({p: difference_se(df, se, p + "s", p + "o") for p in pairs})
    gap = pd.Series({p: RANKS_DESC.index(p[1]) - RANKS_DESC.index(p[0]) for p in pairs})
    lo, hi = premium.idxmin(), premium.idxmax()
    print("\n**Suited premium** (equity of XYs - XYo, 78 rank pairs)\n")
    print(f"- Mean {100 * premium.mean():.2f} pp, range {100 * premium[lo]:.2f} ± {100 * premium_se[lo]:.2f} pp "
          f"({lo}) to {100 * premium[hi]:.2f} ± {100 * premium_se[hi]:.2f} pp ({hi}). "
          f"SE per pair {100 * premium_se.min():.3f}-{100 * premium_se.max():.3f} pp.")
    by_gap = premium.groupby(gap).mean()
    print("- By rank gap (1 = connectors): " + ", ".join(f"{g}: {100 * v:.2f} pp" for g, v in by_gap.items()))

    print("\n**Where the premium comes from** (direct simulation vs a random hand, 1M deals each)\n")
    print("| Hand | P(hero makes a flush) | Equity with a flush | Equity without |\n|---|---|---|---|")
    for seed, label in enumerate(["AKs", "AKo", "32s", "32o"]):
        f = flush_split(label, 1_000_000, seed)
        print(f"| {label} | {f['p_flush']:.1%} | {f['eq_flush']:.1%} | {f['eq_no_flush']:.1%} |")

    _heatmap(eq)


def _heatmap(eq: pd.Series):
    grid = np.array([eq[h] for h in HAND_CLASSES]).reshape(GRID, GRID)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(9, 8.6), dpi=150)
    image = ax.imshow(grid * 100, cmap=cmap, vmin=30, vmax=86)
    for r in range(GRID):
        for c in range(GRID):
            value = grid[r, c] * 100
            color = "white" if value > 58 else INK  # light text only on the dark end of the ramp
            ax.text(c, r - 0.15, HAND_CLASSES[r * GRID + c], ha="center", va="center", fontsize=7.5,
                    color=color, fontweight="bold")
            ax.text(c, r + 0.22, f"{value:.1f}", ha="center", va="center", fontsize=6.5, color=color)
    # 1px white gaps between cells
    ax.set_xticks(np.arange(-0.5, GRID), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Preflop equity vs a random hand (%)   ·   suited above the diagonal, offsuit below",
                 loc="left", fontsize=10, color=INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("equity vs random hand (%)", color=MUTED, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / "heatmap.png"
    fig.savefig(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
