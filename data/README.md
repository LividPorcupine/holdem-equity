# Preflop equity table

Heads-up, all-in preflop equity of each of the 169 hand classes against each other.
Built by `python -m scripts.build_table` (40,000 Monte Carlo samples per cell, seed 2026, 323 s on 24 cores).

## Files
| File | Shape | Contents |
|---|---|---|
| `preflop_equity.csv` | 169 × 169 | `eq[hero][villain]`: hero's equity (0–1). First column `hero` = row labels; header = villain labels. |
| `preflop_equity_long.csv` | 28,561 rows | `hero, villain, equity, std_error, samples` — one row per cell. |
| `preflop_equity.parquet` | 28,561 rows | Same as the long CSV (typed, smaller). |

## Labels and order
Labels: pairs `AA`…`22`, suited `AKs`, offsuit `AKo` (higher rank first).
Rows/columns follow the standard 13×13 grid, row-major from `AA`: diagonal = pairs,
above it = suited, below it = offsuit (`holdem_equity.preflop.HAND_CLASSES`).

## Method
- **Equity** = P(win) + P(tie)/2 over a random 5-card board.
- A cell averages over every concrete combo of both classes that doesn't share a card:
  each sample draws a non-conflicting (hero combo, villain combo) pair uniformly,
  then a uniform board from the remaining 48 cards.
- Only cells above the diagonal are simulated. `eq[j][i] = 1 − eq[i][j]` (same samples), so
  the table is exactly symmetric. Diagonal cells (e.g. AA vs AA) are 0.5 by symmetry (`samples = 0`).
- Standard error per cell: max 0.249 pp, mean 0.229 pp.

## Loading
```python
import pandas as pd
eq = pd.read_csv("data/preflop_equity.csv", index_col="hero")
eq.loc["AKs", "QQ"]          # AKs's equity vs QQ
long = pd.read_parquet("data/preflop_equity.parquet")
```
