"""Exploratory analysis of the preflop table.

Run from the repo root (after build_table):  python -m scripts.eda
Prints the findings as markdown, writes data/equity_vs_random.csv and figures/heatmap.png.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from holdem_equity.preflop import COMBOS, GRID, HAND_CLASSES, RANKS_DESC, equity_vs_random

ROOT = Path(__file__).resolve().parent.parent
# dataviz reference palette: sequential blue ramp, steps 100 -> 700
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED = "#0b0b0b", "#52514e"


def hand_type(label: str) -> str:
    return "pair" if len(label) == 2 else {"s": "suited", "o": "offsuit"}[label[2]]


def main():
    table = pd.read_csv(ROOT / "data" / "preflop_equity.csv", index_col="hero")
    df = equity_vs_random(table).rename("equity").to_frame()
    df["type"] = [hand_type(h) for h in df.index]
    df["combos"] = [len(COMBOS[h]) for h in df.index]
    df = df.sort_values("equity", ascending=False)
    df.insert(0, "rank", range(1, len(df) + 1))
    df.round(6).to_csv(ROOT / "data" / "equity_vs_random.csv", index_label="hand")

    print("## Findings\n")
    print("**Strongest / weakest vs a random hand**\n")
    print("| Rank | Hand | Equity |  | Rank | Hand | Equity |\n|---|---|---|---|---|---|---|")
    for (h1, r1), (h2, r2) in zip(df.head(5).iterrows(), df.tail(5).iterrows()):
        print(f"| {r1['rank']} | {h1} | {r1.equity:.1%} |  | {r2['rank']} | {h2} | {r2.equity:.1%} |")

    weighted = np.average(df.equity, weights=df.combos)
    print(f"\nCombo-weighted average over all 1,326 hands: {weighted:.4f} (must be 0.5: "
          "every matchup's two shares sum to 1).\n")

    print("**Average equity by type** (mean over classes)\n")
    print("| Type | Classes | Mean | Min | Max |\n|---|---|---|---|---|")
    for t, g in df.groupby("type", sort=False):
        print(f"| {t} | {len(g)} | {g.equity.mean():.1%} | {g.equity.min():.1%} ({g.equity.idxmin()}) "
              f"| {g.equity.max():.1%} ({g.equity.idxmax()}) |")

    eq = df.equity
    premium = pd.Series({hi + lo: eq[hi + lo + "s"] - eq[hi + lo + "o"]
                         for i, hi in enumerate(RANKS_DESC) for lo in RANKS_DESC[i + 1:]})
    gap = pd.Series({p: RANKS_DESC.index(p[1]) - RANKS_DESC.index(p[0]) for p in premium.index})
    print("\n**Suited premium** (equity of XYs - XYo, 78 rank pairs)\n")
    print(f"- Mean {100 * premium.mean():.2f} pp, range {100 * premium.min():.2f} pp ({premium.idxmin()}) "
          f"to {100 * premium.max():.2f} pp ({premium.idxmax()}).")
    by_gap = premium.groupby(gap).mean()
    print("- By rank gap (1 = connectors): " + ", ".join(f"{g}: {100 * v:.2f} pp" for g, v in by_gap.items()))

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
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    out = ROOT / "figures" / "heatmap.png"
    fig.savefig(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
