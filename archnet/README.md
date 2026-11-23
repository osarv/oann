# archnet - a network for architectures

archnet learns about architectures while a search runs. It has three parts:

- an **encoder**, which turns genomes into features;
- a **performance head**, which predicts a genome's final loss with an uncertainty and extrapolates training curves;
- a **generator**, which proposes new genomes.

It learns online from every result the search gives it. An `Advisor` puts the three behind five methods. evo/search
calls them (`RunWith`), and any other search can call them the same way:

| method | what it does |
|---|---|
| `Ready() Bool` | says whether it has seen enough to be trusted (`Predictor.Least` results) |
| `Observe(g, rung, curve, loss)` | records that `g` reached `loss` at `rung`, after the losses in `curve` at earlier rungs |
| `Acquire(g) F64` | rates how promising `g` is before training, lower is better: predicted mean − `K` × spread |
| `Forecast(g, curve) F64` | predicts `g`'s final loss from the first points of its curve |
| `Propose(r) G ? NoProposal` | draws a new genome from the generator |

```olang
adv := archnet.Advisor<schema.Genome, encode.SchemaCode, propose.Proposer>(encode.SchemaCode(space), propose.Proposer(space), 2)
c.Screen = 8          # each child is the most promising of 8 candidates (outside the Explore share)
c.Proposals = 0.3     # 30% of children are proposed by the generator
e.RunWith(space, fit, split, adv, rounds)
```

The last argument (2) is the number of rungs before the final one, which is how many curve points the predictor reads.
For graph genomes, use `encode.NetCode` and `propose.NoProposer<neat.Net>`: you get prediction, but no proposals.

| module | what it holds |
|---|---|
| `encode.olang` | `SchemaCode` encodes evo/schema genomes. Choices are one-hot. Numbers are placed in their range, on a log scale for sizes. A repeat is its count, then its first `Slots` instances, then their mean. `NetCode` encodes evo/neat networks by path encoding: the operation sequences on input-to-output paths, plus counts, an operation histogram and widths. |
| `predict.olang` | `Predictor` is an ensemble of small MLPs (`Hidden`, `Members`, `Rate`, `Steps`). Its input is the features, the curve so far (each point with a presence flag) and the rung to predict. The mean over members is the prediction and their spread the uncertainty. Every rung's result is training data, and each training draw takes a random prefix of the curve, so it predicts from any number of points. `Kendall` gives rank correlation. |
| `propose.olang` | `Proposer` is an estimation-of-distribution model (the cross-entropy method) over schema genomes. It keeps a distribution per gene and position, fitted to the best `Elite` share of final results weighted by rank, with `Smooth` of the uniform distribution mixed into every draw. It generates by walking the schema, so proposals keep the good genomes' structure position by position. |
| `archnet.olang` | `Advisor`: the three parts behind the interface above |
| `offline.olang` | `Read` parses a log of finished runs and `Report.Evaluate` measures the predictor on it by k-fold cross-validation (below). |
| `toy.olang` | a synthetic architecture benchmark with a known optimum and deliberately misleading early curves, used by the tests and `bench/archsearch` |

## How a search uses it

- **Screening.** Once the advisor is `Ready`, each child is the candidate with the lowest `Acquire` out of
  `Config.Screen`. A genome already evaluated at the full budget is passed over. `Config.Explore` (25% by default) of
  children skip screening entirely, so the population never follows the predictor alone.
- **Proposals.** `Config.Proposals` of children come from `Propose` instead of mutation, and are screened the same way.
  If the generator has nothing to give, the child is a mutation instead.
- **Curve-based early stopping.** With `Config.Forecast` set, successive halving ranks candidates between rungs by the
  final loss the predictor forecasts from their curves, rather than by their loss so far.
- **Observation.** The advisor sees every result at every rung. Its state is not part of evo's checkpoint, so after
  `Resume`, call `e.Replay(adv)`. This re-observes the history, but not the candidates halving dropped.

The predictor's own settings are plain fields, set before its first observation, so a search can tune them too:
`Hidden` (a Vec of widths), `Members`, `Rate`, `Steps` and `Least`. The same holds for the generator's `Elite` and
`Smooth`, and for `Advisor.K`.

## Measuring the predictor on your own runs

`make archeval ARGS="LOG 0,2,5,10"` cross-validates the predictor on a log of finished runs. For each m, it prints
Kendall's tau of the predicted against the true final loss when the predictor is given the first m curve points.
Beside it, the tau of the m-th point alone, which is the loss reached so far. Other arguments: folds (5), seed, and
the predictor's hidden width and members.

The log has one run per line, with three tab-separated fields:

```
loss <TAB> curve <TAB> features
```

- `loss`: the final validation loss.
- `curve`: the validation losses at the curve's points, space-separated, in order. The last point is at the full
  budget, and every run has the same number of points.
- `features`: the genome's features, space-separated, as `encode` writes them (`encode.SchemaCode(space).Encode(g)`).
  May be empty, in which case the predictor ranks from the curve alone.

Lines that are empty or start with `#` are skipped. To read the tau at a given training time, pick m as the curve
point at that time: with 20 points over 3600 s, m = 2 is 360 s and m = 5 is 900 s.

## Measured on archnet/toy

`make archsearch ARGS="60 5"`: 60 rounds × 5 seeds, Regularized(32, 8), 9 children a round through rungs 1, 3, 9.

**How well the predictor ranks** 300 held-out random genomes (Kendall tau), after 300 observations:

| predictor input | tau |
|---|---|
| genome only | 0.57 |
| + 1 curve point | 0.77 |
| + 2 curve points | 0.82 |
| for comparison: the raw loss at budget 1 alone | 0.75 |

Genome-only ranking is data-limited. Trained on 1000 observations instead of 300 (another seed), it reaches 0.75.

**Mean final gap to the optimum:**

| arm | mean final gap |
|---|---|
| evolution | 0.19 |
| with screening | 0.24 |
| with screening and proposals | 0.19 |
| with screening and forecast halving | 0.35 |

At these budgets (about 800 observations) the predictor saves no compute, and forecast halving costs quality.
`Forecast` is therefore off by default. On 8 seeds the picture is the same: evolution 0.20, screening 0.23. Every arm
stalls on the toy's plateaus, where the best genome is 0.15, 0.30 or 0.45 above the optimum. Each plateau is a
depth change that shifts every layer's position in the operation table, a trap no model of single moves gets out of.
On mutants of the final population the predictor still ranks with tau 0.54 to 0.72.

## Measured on a chain landscape

`toy.Chain(n)` is a search where the order of evaluation matters. It has n choice genes of 4 options each. Its loss
sums one cost per gene and one per pair of neighbours (an NK landscape with K = 1), and `OptimumOf` gives the optimum
exactly. Most mutations there are neutral or bad. `ARCH_CHAIN=n make archsearch ARGS="60 8"` runs it, with the same
search as above, over 8 seeds:

| arm | n = 16: mean final gap | seeds within 0.1 | n = 32: mean final gap |
|---|---|---|---|
| evolution | 0.169 | 3 of 8 | 0.88 |
| with screening | 0.065 | 7 of 8 | 0.54 |
| with screening and proposals | 0.061 | 7 of 8 | 0.54 |
| with screening and forecast halving | 0.118 | 6 of 8 | 0.67 |

Here the predictor earns its keep. With screening, the seeds that came within 0.1 of the optimum did so after a mean
budget of 812, against 1101 for the three seeds of evolution that got there at all. The held-out tau on the genome
alone is 0.66 (n = 16) and 0.45 (n = 32).

**The generator:**

- Fitted on 200 random genomes, its proposals have a true loss below 0.75 of the random mean (tested).
- In the search, 48% of proposals beat the median mutated child.
- 452 of the 455 genomes evaluated are distinct.
