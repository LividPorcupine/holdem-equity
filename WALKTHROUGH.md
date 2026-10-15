# holdem-equity: walkthrough

This is written for anyone new to the project. You don't need to know poker maths or much Python: every term is explained where it first appears, and the Python features used are described in plain language. It goes through every file in the order data flows through the code. Then it covers the maths, how the numbers were checked, and a two-minute summary you can say out loud.

**Poker basics used here:** each player gets 2 private cards (*hole cards*). Five shared cards (*the board*) are dealt face up: three at once (*the flop*), then one (*the turn*), then one more (*the river*). Each player makes the best 5-card hand from their 2 cards plus the 5 on the board, and the better hand wins. "All-in" means no more betting happens, so only the cards decide the result.

**Terms used throughout:**
- **Hero / villain:** the two players. Hero is the hand we compute equity for; villain is the opponent (a known hand, or "random").
- **Preflop:** before any board cards are dealt.
- **Suited / offsuit:** both hole cards share a suit (`AKs`) or not (`AKo`). Notation: `Ah` = ace of hearts; suits are c, d, h, s; `T` = ten.
- **C(n, k):** the number of ways to choose k items from n, ignoring order. C(48, 5) = 1,712,304.
- **Monte Carlo:** estimating an answer by averaging many random trials.
- **Standard error (SE):** how far a Monte Carlo estimate typically is from the true value.
- **pp (percentage points):** the difference between two percentages; 46% to 47% is 1 pp.
- **numpy:** Python's library for fast maths on whole arrays of numbers. **Vectorised** code does one operation on a whole array at once instead of looping element by element in Python, which is much faster.
- **pandas:** a library for labelled tables (DataFrames), like a spreadsheet in code. **matplotlib** draws the charts. **treys** is an open-source poker hand evaluator. **pytest** runs the tests.

## The big picture

The question this project answers: **two players are all-in. What share of the pot does each one win on average?** That share is called *equity*:

> equity = P(win) + P(tie) / 2

On a tie the pot is split, so a tie counts as half a win.

```
 "AhKh" ──► cards.py ──► ints 0..51 ──► equity.py ──► deals (hands + boards) ──► evaluate.py ──► hand ranks
                                          │  exact: enumerate every board                         │
                                          │  Monte Carlo: sample random boards                    ▼
                                          └──────────────── compare ranks: win / tie / loss ◄──────┘
                                                                        │
 preflop.py: repeat for all 169 × 169 hand classes ──► data/*.csv ──► scripts/eda.py ──► findings + heatmap
```

There are two ways to get the answer:
- **Exact:** try every possible board and count wins. With no board cards dealt yet, that is C(48, 5) = 1,712,304 boards.
- **Monte Carlo:** try n random boards and average. This is faster when there are too many cases to enumerate, and the error shrinks like 1/√n.

---

## File by file

### `holdem_equity/cards.py`: cards as small integers
**What it does:** turns text like `"Ah"` into an int and back, and builds the deck of remaining cards.

**Why it's designed this way:** a card is `rank * 4 + suit`, a number from 0 to 51. Rank 0 is "2" and rank 12 is "A". Suits 0–3 are c, d, h, s. Small integers can index straight into numpy arrays, which the fast code depends on. The alternative would be a `Card` object with rank and suit fields, but numpy can't do fast whole-array maths on a collection of objects.

- `parse_card("Ah")` returns 50. It validates the input with a regular expression and raises `ValueError` on bad input such as `"1h"` or `"Ax"`. The rank is case-insensitive.
- `parse_cards("AhKh")` returns `[50, 46]`. It splits the string into 2-character chunks and rejects duplicates.
- `card_str` and `cards_str` go the other way, from int back to text.
- `remaining_deck(*dealt)` returns every card not already dealt, as a sorted numpy array. It raises if one card appears twice, for example if both players are given A♥.

**Python idioms:**
- `RANKS.index(c)` returns the position of character `c` in the string `RANKS`.
- `*dealt` in a function definition collects any number of arguments into a tuple (a fixed list), so `remaining_deck(a, b, c)` works with as many card lists as you pass.
- `np.flatnonzero(mask)` returns the indices where a boolean array is `True`.

### `holdem_equity/evaluate.py`: how strong is a hand?
**What it does:** given 5–7 cards, it returns the rank of the best 5-card hand. Rank 1 is a royal flush and rank 7462 is 7-5-4-3-2 offsuit. Lower is stronger.

**Why it's designed this way:** the project uses `treys` but hides it behind its own interface, so this is the **only file that imports treys**. Swapping in a different evaluator later means changing only this file.

The file has two entry points:
- `evaluate(cards)` handles one hand. It converts our ints to treys ints and calls `treys.Evaluator`. It is simple and is the reference that the tests trust.
- `evaluate_batch(hands)` handles an `(n, 7)` array of hands at once. It returns exactly the same answers about 8× faster, measured at 1.20M hands/s against 0.15M/s (`python -m scripts.benchmark` reproduces this).

**How the batch version works (the clever part):** treys uses Cactus Kev's trick. Each rank is assigned a prime (2→2, 3→3, 4→5, …, A→41), and a 5-card hand is identified by the **product** of its five primes. Multiplication doesn't care about order, so `AKQJT` and `TJQKA` give the same number. Unique factorisation also means different rank multisets give different products. treys keeps two dictionaries mapping product → rank, one for flushes and one for everything else.

`_evaluate_chunk` does the same thing for a whole batch at once:
1. `TREYS_CARDS[hands]` converts every card to its treys int. This is numpy *fancy indexing*: an array used as an index.
2. `cards[:, _SUBSETS[7]]` builds all 21 five-card subsets of each 7-card hand, giving shape `(n, 21, 5)`.
3. `five & 0xFF` takes the prime stored in the low byte of each treys card. `np.prod(..., axis=2)` multiplies the five primes.
4. `np.bitwise_and.reduce(five & 0xF000, axis=2) != 0` detects a flush. Bits 12–15 hold the suit as a one-hot flag, so the AND of all five is non-zero only if they share a suit.
5. `np.searchsorted` looks every product up in treys' tables, stored as sorted arrays, in one call. The flush table is only searched where there is a flush.
6. `.min(axis=1)` keeps the best of the 21 subsets.

`_CHUNK = 50_000` processes hands in slices so memory stays around 100 MB.

**Python idioms:**
- `np.where(cond, a, b)` picks from `a` where `cond` is true and from `b` elsewhere, for every element at once (an element-by-element if/else).
- `axis=2` means "reduce along the 3rd dimension".
- Module-level constants like `_FLUSH_KEYS` are computed once, when the file is first imported, and then reused by every call.
- A leading `_` means "private by convention". Python doesn't enforce it.

### `holdem_equity/equity.py`: exact and Monte Carlo equity
**`EquityResult`** is a `@dataclass(frozen=True)`: a small class that just holds named values. Python writes the constructor and comparison code for you, and `frozen=True` means the values can't be changed after creation. It holds `equity`, `win`, `tie`, `std_error` (0 for exact results) and `showdowns`, the number of deals counted.

**`showdown_outcomes(hero, villain_cards, boards)`** is the core comparison. It stacks hero's 2 cards onto each board (`np.hstack`) and evaluates the whole batch. It does the same for the villain, then returns 1 / 0.5 / 0 per deal. `np.broadcast_to` lets one fixed hero hand stand in for n identical rows without copying it n times.

**`exact_equity(hero, villain=None, board=())`:**
- It works out the number of showdowns first. If that exceeds `MAX_EXACT_SHOWDOWNS` (20M, about 30 s), it raises an error that points you to Monte Carlo. Preflop against a *random* hand would be C(50,2) · C(48,5) ≈ 2.1 billion showdowns.
- `_all_combos` builds every runout as an `(n, k)` int8 array. It uses `itertools.combinations` streamed into `np.fromiter`, so no huge Python list is created.
- With a random villain it loops over every villain hand, which is equally likely, and enumerates the runouts for each. Every (villain, board) pair is equally likely, so plain counting gives the right answer.
- `_tally` keeps running totals of `[wins, ties, showdowns]`.

**`monte_carlo_equity(hero, villain=None, board=(), samples=100_000, seed=None)`:**
- `np.random.default_rng(seed)` creates a seeded generator, so the same seed gives the same answer. A test checks this.
- `rng.permuted(np.tile(deck, (n, 1)), axis=1)` shuffles each of n copies of the remaining deck independently. The first few cards of each row become that deal's villain cards (if random) and board.
- It returns `equity = mean(x)` and `std_error = std(x) / √n`. See [the maths](#why-monte-carlo-error-falls-like-1n).

### `holdem_equity/preflop.py`: 169 classes and the big table
**Hand classes:** before the flop, only "same suit or not" matters, so the 1,326 two-card hands fall into 169 classes:

| Class type | Count | Combos each | Why |
|---|---|---|---|
| Pair (AA) | 13 | **6** | choose 2 of the 4 aces: C(4,2) = 6 |
| Suited (AKs) | 78 | **4** | one combo per suit |
| Offsuit (AKo) | 78 | **12** | 4 suits for the A × 3 different suits for the K |

13·6 + 78·4 + 78·12 = 78 + 312 + 936 = **1,326 = C(52,2)**. A test checks that every hand appears exactly once.

- `grid_label(row, col)` gives the standard 13×13 grid: pairs on the diagonal, suited above it, offsuit below. `HAND_CLASSES` lists all 169 in that order.
- `class_combos(label)` returns every concrete hand in a class as a `(k, 2)` array.
- `_masks` represents each combo as a 52-bit set of cards (`1 << card`). Two hands share a card exactly when `mask_a & mask_b != 0`. `_valid_pairs` uses this to list every (hero combo, villain combo) pair with no shared card. For example, AA vs AA has only 6 valid pairs, because each AA combo avoids only its complementary AA combo.

**`class_vs_class(hero, villain, samples, rng)`:**
1. It picks one valid combo pair per sample, uniformly, which averages over suits fairly.
2. It deals a board using a *random-keys* trick. Every card gets a random key in [0, 1), and the 4 dealt cards get key 2.0 so they always sort last. The 5 smallest keys (`np.argpartition`) are then a uniform random 5-card board from the other 48 cards. `argpartition` is faster than a full sort because it only finds the smallest 5.
3. It evaluates all samples with `showdown_outcomes` and returns the mean and the standard error.

**`exact_class_vs_class(hero, villain)`** gives the exact class average with no sampling. Renaming the suits (say hearts ↔ spades everywhere) doesn't change anyone's equity, so `_suit_pattern` maps each combo pair to the smallest relabelling of its suits. Pairs with the same pattern have the same equity. Each pattern is enumerated once over all 1,712,304 boards and weighted by how many combo pairs share it. AKs vs QQ has 24 valid pairs but only 2 patterns: the QQ either holds a card of the AK's suit or it doesn't. `Counter` (from `collections`) counts how often each pattern occurs. A test compares this with Monte Carlo.

**`build_table(samples=40_000, seed=2026)`:**
- Only cells above the diagonal are simulated (14,196 of them). The mirror cell is filled as `eq[j][i] = 1 − eq[i][j]`: heads-up, the two players' shares of the pot always add to 1. That makes the table exactly symmetric by construction. Diagonal cells such as AA vs AA are exactly 0.5, because neither side has an edge.
- `multiprocessing.Pool` spreads the rows across CPU cores. Each row gets its own seed from `np.random.SeedSequence(seed).spawn(169)`, so the results are identical no matter how many cores run it or in what order rows finish (`imap_unordered`).
- It returns two pandas DataFrames, equity and standard error, labelled by class.

**`equity_vs_random(table, se)`:** each class's equity against a random hand, worked out from the table without new simulation, plus its standard error. See [the weighting](#equity-vs-a-random-hand-from-the-table).

**`to_long`** reshapes the 169×169 matrix into one row per cell (`DataFrame.stack()`) for the long CSV and parquet files.

### `holdem_equity/cli.py` and `__main__.py`
`python -m holdem_equity AhKh QsQd [--board Td9h2c] [--exact | --samples N] [--seed S]`. `__main__.py` is what `python -m holdem_equity` runs. It just calls `cli.main()`.

- `argparse` defines the arguments. `add_mutually_exclusive_group` stops you from passing both `--exact` and `--samples`. The second hand can be the word `random`.
- Any `ValueError` from parsing or from the equity code is turned into `parser.error(...)`, which prints usage and exits with code 2. That is the standard convention for bad command-line input.
- The ± is printed as ASCII `+/-` because Windows consoles garble `±`.

### `scripts/`: things you run once
These are run as modules from the repo root, for example `python -m scripts.build_table`, so that `holdem_equity` can be imported.
- **`build_table.py`** builds the table (323 s on 24 cores), writes `data/preflop_equity.csv`, `_long.csv` and `.parquet`, and prints the checks: max standard error, symmetry, and the extremes.
- **`eda.py`** loads the table and its standard errors, then does the analysis with pandas: the ranking against a random hand (also saved to `data/equity_vs_random.csv`), averages by type with `groupby`, the suited premium for each of the 78 rank pairs, and the heatmap. It prints the findings as markdown, with a standard error on each derived number.
  - `difference_se(a, b)` gives the SE of a difference such as 72o − 32o or AKs − AKo. The two rows share one simulated cell, and since eq[a][b] = 1 − eq[b][a] that cell pushes both rows' equities apart, so its variance is added, not cancelled.
  - `flush_split(label, ...)` simulates a hand against random hands directly and splits the result by whether hero ends with a flush. It backs the explanation of the suited premium.
- **`benchmark.py`** ranks the same random 7-card hands with `treys` one at a time and with `evaluate_batch`, checks that they agree and prints both speeds.
- **`convergence.py`** handles 3 matchups × 9 sample sizes × 30 seeds and plots mean |MC − exact| on log-log axes against the theoretical curve. `_spread` nudges the end-of-line labels apart so they don't overlap.

`if __name__ == "__main__": main()` means "run `main()` only when this file is run directly, not when it's imported". It is also *required* with `multiprocessing` on Windows: each worker process re-imports the script, and without that guard each worker would start its own pool.

### `tests/` (61 tests, `pytest`)
pytest finds every function named `test_*` and runs it. A plain `assert` is the whole assertion API. `@pytest.mark.parametrize` runs one test for each set of inputs, and `pytest.raises` checks that an exception is thrown.

| File | Checks |
|---|---|
| `test_cards.py` | round-trips, bad input rejected, deck removal, card conflicts |
| `test_evaluate.py` | royal flush = rank 1, worst hand = 7462, batch == treys on 3,000 random 5/6/7-card hands |
| `test_equity.py` | exact values within expected ranges and pinned to 1e-12, river win/split, showdown counts, refusal of huge enumerations, MC within 4 SE of exact, seeded reproducibility, SE falls 4× for 16× the samples, numpy-array hands accepted |
| `test_preflop.py` | 169 classes, 6/4/12 combos, 1,326 distinct hands, combo-pair counts, mirror matchup ≈ 0.5, exact AKs vs QQ class average (0.460485) matched by Monte Carlo within 4 SE, and the acceptance checks on the shipped table (shape, symmetry, AA top ≈85%, 32o bottom ≈32%, 72o ≈35%, max SE < 0.3 pp). A direct 1M-sample simulation against a random hand must agree with the table for 32o, 72o, AA and 22 (\|z\| < 4) |
| `test_eda.py` | the suited-premium explanation: suited hands make far more flushes and have over 90% equity with them, and a flush is worth more to 32 than to AK |
| `test_cli.py` | Monte Carlo, exact and random modes; bad input exits with code 2; `python -m holdem_equity` works |

`@cache` (from `functools`) on `exact_preflop` means each 3-second exact matchup is computed once and shared by several tests.

---

## The maths

### Exact enumeration counts
- No board yet: 52 − 4 = 48 unseen cards, so C(48, 5) = 1,712,304 boards.
- Flop given: C(45, 2) = 990 turn+river pairs. Turn given: 44 rivers.
- Against a random hand on the flop: C(47, 2) = 1,081 villain hands × C(45, 2) = 990 runouts = 1,070,190 showdowns.

### Why Monte Carlo error falls like 1/√n
Each deal gives an outcome x ∈ {1, 0.5, 0} with mean μ (the true equity) and standard deviation σ. The deals are independent, so the average of n of them has

> Var(x̄) = σ² / n  ⇒  SD(x̄) = σ / √n

This is the **standard error**. The code estimates σ from the samples as `std(x, ddof=1)`, where `ddof=1` divides by n − 1, the unbiased version.

Consequences:
- Halving the error costs **4×** the samples. Ten times more accuracy costs 100× the samples.
- On a log-log plot, error against n is a straight line with slope −½. That is what `figures/convergence.png` shows across four decades of n.
- By the central limit theorem x̄ − μ is approximately normal, and for a normal variable E|Z| = σ√(2/π). That gives the dashed theory curve **E|error| = √(2/π)·σ/√n**.
- σ for one deal comes from the exact results: E[x²] = P(win)·1² + P(tie)·0.5², so σ² = P(win) + P(tie)/4 − μ². For AA vs KK, σ = 0.389. For the near coin-flips, σ = 0.498, close to the maximum of 0.5.

**Samples per table cell:** an outcome in [0, 1] has variance at most ¼, so σ ≤ 0.5 and SE ≤ 0.5/√n. This bound is known before any sampling. For SE < 0.3 pp = 0.003 we need n > (0.5/0.003)² ≈ 27,778. The build uses **40,000**, so the worst case is 0.25 pp. The measured maximum was 0.249 pp.

**Reading a single run:** at 100k samples the SE is about 0.16 pp, so a 95% interval is ±1.96 SE ≈ ±0.31 pp. The README's three runs miss the exact value by 0.6, 2.1 and 1.8 SE. Being outside 1.96 SE once in three runs is unremarkable, because it happens about 1 run in 20. That is why the convergence plot averages 30 seeds.

### Equity vs a random hand, from the table
Deal hero a random combo c of class i, and villain a random 2-card hand d from the 50 remaining cards. Every hero combo leaves exactly C(50,2) = 1,225 villain hands, so every *valid* (c, d) pair is equally likely. Class j therefore appears with probability proportional to `counts[i, j]`, the number of valid (c, d) pairs with c ∈ i and d ∈ j:

> eq_vs_random(i) = Σⱼ counts[i,j] · eq[i,j] / Σⱼ counts[i,j]

The weights matter. For example, AA has fewer valid pairs against other Ax hands, because the aces block them.

**Standard error.** Within row i, every cell comes from a separate simulation, so the cells are independent and their variances add with squared weights:

> SE(i) = √(Σⱼ counts[i,j]² · se[i,j]²) / Σⱼ counts[i,j]

That comes to about 0.02 pp per hand, which is why 72o − 32o = 2.26 ± 0.03 pp is clearly real.

**The check that can fail:** simulate one combo directly against random hands (1M samples) and compare it with the formula as a z-score. For 32o, 72o, AA and 22 the |z| values are 1.4, 1.4, 1.7 and 0.9. The test fails if any reaches 4.

**The check that can't fail:** the combo-weighted average over all 1,326 hands is exactly 0.5. That is true by construction (eq[j][i] = 1 − eq[i][j]): any table filled that way averages to 0.5, even a random one. So it is a consistency identity, not a test of the numbers.

---

## How the numbers were checked
| Check | Result |
|---|---|
| Batch evaluator vs `treys.Evaluator` | identical on 200k random hands while developing, plus 9k in the tests |
| A♥A♠ vs K♦K♣ exact vs a plain `treys.Evaluator` loop over all 1,712,304 boards | identical: 0.8125548967940273. Both use treys' lookup tables, so this checks the vectorisation, not treys itself |
| A♥A♠ vs K♦K♣ vs published figures | [PokerSkill](https://www.pokerskill.com/blog/aa-vs-kk/): aces win 81.1% with no shared suits, equity 81.3–82.6% depending on suits. Ours: win 81.06%, equity 81.26% |
| AK vs 22 vs published figures | [PokerStrategy](https://www.pokerstrategy.com/strategy/1776): 47.35%. Ours: AKo vs 22 exact class average 47.35% (`exact_class_vs_class`). The specific A♥K♦ vs 2♠2♣ is 46.96%, because suits matter |
| Monte Carlo bias | 10 × 1M fresh-seed samples per matchup: pooled z-scores 0.98, 0.55, 0.45. No bias |
| AA / 72o / 32o vs random | 85.19% / 34.58% / 32.32%, against expected values of ≈85 / ≈35 / ≈32. Each also matches a direct 1M-sample simulation within \|z\| ≤ 1.3 |

---

## Explain it in 2 minutes
> "I chose this project and learned it end to end; it was built with an AI coding agent. It's a poker equity calculator in Python. Equity is your expected share of the pot when everyone's all-in: P(win) plus half of P(tie).
>
> There are two engines. The exact one enumerates every possible board. Preflop that's 1.7 million boards, and it runs in about 3 seconds because hand evaluation is vectorised with numpy. The evaluator uses treys' Cactus Kev lookup tables, where a hand is identified by the product of prime numbers assigned to its ranks, and looks up whole batches of hands at once. It's 8 times faster than calling treys one hand at a time, and it's tested to give identical answers.
>
> The second engine is Monte Carlo: deal random boards and average. Its standard error is σ/√n, so 100 times the samples buys 10 times the accuracy. The log-log convergence plot, averaged over 30 seeds, sits on the theoretical curve. That's the same validation you'd run on a Monte Carlo pricer: check it against a case with an exact answer.
>
> Then Monte Carlo builds the full 169-by-169 table of every starting-hand class against every other. It uses 40,000 samples per cell, because an outcome between 0 and 1 caps the standard error at 0.25 points, and it runs in parallel across cores with reproducible seeds. With pandas: AA is best at 85% against a random hand, the worst hand is actually 32 offsuit rather than 72 (a 2.3-point gap with a standard error of 0.03), and being suited is worth about 2.75 points. A direct simulation agrees with the table within 2 standard errors. That table feeds the next project, a push/fold solver using counterfactual regret minimisation."
