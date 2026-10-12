# holdem-equity

Texas Hold'em equity calculator in Python, exact and Monte Carlo, plus a 169×169 preflop equity table with an analysis of it.

## Results

**Exact preflop all-in equity** (all 1,712,304 boards enumerated, ~3 s each):

| Matchup | Exact | Monte Carlo, 100k samples (seed 1) | Error |
|---|---|---|---|
| A♥A♠ vs K♦K♣ | **81.26%** | 81.33% ± 0.12 | +0.08 pp |
| A♥K♥ vs Q♠Q♦ | **46.21%** | 45.88% ± 0.16 | −0.33 pp |
| A♥K♦ vs 2♠2♣ | **46.96%** | 46.68% ± 0.16 | −0.28 pp |

Every exact value was cross-checked against a plain `treys` enumeration and published figures. Details are in [WALKTHROUGH.md](WALKTHROUGH.md#how-the-numbers-were-checked).

**Monte Carlo error falls like 1/√n.** Each point is the mean |MC − exact| over 30 seeds. The dashed lines are the theoretical √(2/π)·σ/√n.

![Convergence of Monte Carlo error](figures/convergence.png)

**Every starting hand against a random hand.** The values come from the 169×169 table. Suited hands are above the diagonal and offsuit hands below it.

![Preflop equity heatmap](figures/heatmap.png)

### Findings
- **Best and worst hands:** AA is the best starting hand at 85.2% against a random hand, followed by KK at 82.4% and QQ at 79.9%. The worst is **32o at 32.3%**, not the famous 72o (34.6%, 5th from bottom). Both of 32o's cards are the lowest ranks, so its pairs and high cards lose to almost everything. Its extra straight chances (A-5, 2-6) don't make up for that, while 72o's 7 wins more high-card and pair showdowns.
- **Pairs:** pairs average 68.6% against a random hand. Even 22 is above 50% (50.3%).
- **Suited vs offsuit:** suited hands average 50.9% and offsuit hands 48.1%.
- **Suited premium:** being suited is worth **+2.75 pp on average**. It ranges from 1.77 pp (AK) to 3.65 pp (32). Weak hands gain the most, probably because a flush lifts a hand that otherwise rarely wins.
- **Sanity check:** the combo-weighted average across all 1,326 hands is exactly 0.5000, as it must be.

## Run

```
pip install -r requirements.txt

python -m holdem_equity AhKh QsQd                     # Monte Carlo, 100k samples
python -m holdem_equity AhKh QsQd --board Td9h2c --exact
python -m holdem_equity AhAs random --seed 1          # vs a random hand
```

```
AhKh vs QsQd, board: (none)  [Monte Carlo, 100,000 samples, seed 1]
  AhKh    45.88% +/- 0.16   (win 45.70%, tie 0.37%)
  QsQd    54.12% +/- 0.16
```

Rebuild the data and figures:

```
python -m scripts.build_table    # data/preflop_equity*.csv/.parquet (~5 min on 24 cores)
python -m scripts.eda            # data/equity_vs_random.csv, figures/heatmap.png, findings
python -m scripts.convergence    # figures/convergence.png (~1 min)
```

## Test

```
pytest
```

There are 54 tests. They cover card parsing, the evaluator against `treys`, exact values, Monte Carlo tolerance and reproducibility, the table's symmetry and extremes, and the CLI.

## Data

The preflop table's format and method are documented in [data/README.md](data/README.md). It is the input for a planned heads-up push/fold CFR solver.

## Stack

Python 3.11+ · numpy (vectorised evaluation and sampling) · pandas (table + EDA) · matplotlib · `treys` (hand-rank lookup tables) · multiprocessing · pytest

Full explanation: [WALKTHROUGH.md](WALKTHROUGH.md) · MIT licence
