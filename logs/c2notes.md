**As built (C2, 2026-11-15).** `dyn/bridge.olang`, `models/hybrid.olang`, `bench/hybrid.olang` (`make hybridbench`).
Where the build departs from the sketch above (`dyn.Settle(g, system, input, grad)`), and why:

1. **A `bridge.Bridge<T>(s, inp, out, how)` is the operation** (it satisfies `graph.Function`): its input is population
   inp's drive, its output population out's activities at the free fixed point, its backward `learn.Learner.ImplicitGrad`
   by `how` - the input's gradient to the graph, the system's parameters' written into `s.ParamGrads()`, which an
   optimizer of its own steps (the system's region is not the graph's). `bridge.Apply(g, b, settled, x, guess)` records
   it. The input may have leading axes (`[batch, t, size]` is batch x t rows of the system, at most its batch; a short
   batch's rows past it are driven by zero). Generic over T so the checks run it in F64 (`Run`, `RunFrom`, `Grad`); in a
   graph it is F32, as `Custom` is.
2. **The amortizer's guess is a second input**: the starting potentials of every relaxing population (every one but
   the inputs, in the order made: `b.StateSize` a row), their activities rho of them; without it a settle starts at rest.
   Its gradient is zero (at a fixed point the output does not depend on the start). The amortizer is trained by
   regression on the fixed point: `bridge.Settled(b)` is an operation reading the bridge's output (so it runs after the
   settle) whose value is the potentials reached and whose gradient is zero, and `Apply` gives `g.Mse(guess, settled)`;
   `bridge.Detach` (a copy, zero gradient) keeps that loss from reaching the amortizer's input. The whole loss's
   gradient is then the task's plus the amortizer's regression with the fixed point held - a stopped gradient, so the
   graph's gradient is not the central differences of a loss that includes the regression (tested separately: the
   task's loss against its differences, the regression's against its differences in the amortizer's weights).
3. **dz is scaled to a largest element of 1** before the learner sees it and the gradients scaled back: the exact
   adjoint's tolerance and EP's nudge are absolute, and a mean over 1 024 rows gives elements near 1e-4, which a
   tolerance of 1e-3 would read as solved in zero sweeps and EP's nudge would drown in the settle's tolerance. A
   backward the learner refuses (an adjoint or nudged phase that does not settle) writes zero gradients and counts in
   `Refused`; a forward settle that ends outside its budget is used as reached and counts in `Unsettled`.
4. **JFB is not the gradient of a recurrent settle**, so "against central differences" holds for it only where
   nothing feeds back into the output (x -> y, where it is exact, 1e-10); with feedback its cosine with the gradient is
   stated (0.98 on the test system). Exact and EP are checked against central differences on a recurrent system of
   every reciprocal wiring.
5. **Not pinned allocation-free**: a step through a `Custom` allocates (B1's note: its tensors are built where the graph
   lives at each call); the bridge itself allocates nothing per call (its staging is laid out at construction).
6. **The hybrid** (`models/hybrid.olang`): GPT-2's stream (embedding, learned positions, `layers.AttentionBlock`s), then
   the settle block - the stream normalized drives Dim linear inputs of a system of Units tanh units (a symmetric
   lateral block) and Dim tanh outputs (symmetric with the units), certified at Kappa; its outputs, projected, are added
   back; a final norm and the tied head. The amortizer is a tanh perceptron Dim -> 2 Dim -> StateSize on the detached
   normalized stream. `hybrid.Step` steps both regions (the graph's by nn/train's `LmStep`, then the system's gradient
   clipped, its optimizer stepped and `Restrain(Kappa)`, so every settle stays certified). It is made in pieces the
   program holds - system, bridge, `Settled`, `Detach`, graph - since each refers to the one before
   (repro/learnerarg.olang: a Learner, so a Bridge, cannot even be made in a helper taking the system by reference).

Measured (tests in `dyn/bridge.olang`, `models/hybrid.olang`): the op's backward against central differences of its
output (F64, a recurrent system of a sparse, a low-rank, a tiled and a symmetric block, two leading axes): exact 1.0e-10
in the parameters and 6.1e-10 in the input, EP(1e-3) 3.3e-7 and 1.6e-7; JFB on x -> y 1e-10 (cosine 0.98 with feedback);
the gradients scale with dz to 1e-9 in every mode; a settle started at its fixed point takes at most one sweep and ends
where a cold one does (1e-12). Through an F32 graph (Linear -> settle -> Linear -> Mse) against F32 central differences
(step 1e-2) of the graph's loss, the graph's parameters / the system's: exact 8.1e-5 / 1.8e-4, EP(0.01) 2.9e-4 /
2.8e-4, JFB (no feedback) 9.1e-6 / 4.8e-5; from an amortizer's guess the same to within 1e-5, and the amortizer's
weights against its regression's differences 3.3e-6. The hybrid trains on a cycle (4 sequences of 8, width 16) from ln
4 to 0.006 in 150 steps by every gradient, cold and amortized, 5-6 sweeps a settle cold and 3 amortized.

**M14** (`make hybridbench ARGS="1500 1 128 exact 1e-3 KAPPA 250"`, tiny Shakespeare characters, 2 layers of 64 in 4
heads, context 64, 16 sequences - 1 024 rows a settle - a settle block of 128 units, 192 units settling, exact implicit
gradient, forward tolerance 1e-3, AdamW 1e-3 warmed over 100 steps, cosine to 1e-4, both regions, one thread, load
14-19 on 4 cores; validation sweeps on the same 10 batches at each report, weights fixed). Certified at kappa 0.9:

| step | valid loss, cold | amortized | sweeps a settle, cold | amortized | adjoint sweeps | ms a step, cold | amortized |
|---|---|---|---|---|---|---|---|
| 250 | 2.4968 | 2.4956 | 5.0 | 4.0 | 3.1 | 214 | 202 |
| 500 | 2.3502 | 2.3506 | 5.0 | 4.1 | 3.3 | 197 | 200 |
| 750 | 2.2694 | 2.2694 | 5.1 | 4.9 | 4.0 | 234 | 205 |
| 1000 | 2.1856 | 2.1873 | 6.0 | 4.0 | 4.0 | 251 | 199 |
| 1250 | 2.1436 | 2.1457 | 6.0 | 4.4 | 4.1 | 245 | 207 |
| 1500 | 2.1182 | 2.1198 | 6.0 | 4.0 | 4.1 | 209 | 206 |

The amortizer saves a third of the forward sweeps once trained (6 to 4 at 1 500 steps; its regression loss 0.0021 per
element), at no cost in the loss (within 0.002 throughout) - but at kappa 0.9 the certificate is loose (the row-sum
bound; the spectral radius of these blocks is far below it), a cold settle needs only 5-6 sweeps, and the step's time is
the transformer's and the adjoint's, so the saving does not show in ms a step at this load.

With the recurrent blocks held at kappa 4 instead (`... exact 1e-3 4 250`: not certified - the row-sum bound is 4, the
settles still converge, 40-50 sweeps; the exact adjoint gave up within 400 sweeps on 4 backward passes of 1 500 in each
run, `Refused`, their gradients zero):

| step | valid loss, cold | amortized | sweeps a settle, cold | amortized | adjoint sweeps, cold | amortized |
|---|---|---|---|---|---|---|
| 250 | 2.5108 | 2.5104 | 23.9 | 18.7 | 11.6 | 11.7 |
| 500 | 2.3513 | 2.3546 | 32.4 | 30.9 | 27.5 | 27.3 |
| 750 | 2.2765 | 2.2805 | 37.9 | 38.3 | 40.6 | 39.3 |
| 1000 | 2.1909 | 2.1904 | 37.0 | 41.9 | 32.5 | 41.3 |
| 1250 | 2.1440 | 2.1414 | 45.2 | 40.3 | 34.8 | 36.8 |
| 1500 | 2.1134 | 2.1124 | 39.6 | 53.5 | 36.3 | 33.3 |

Here the amortizer saves nothing past the first few hundred steps: its regression loss stays at 0.022 (ten times
kappa 0.9's), and a settle's sweeps go as ln(start error / tol) / ln(1/q) with q near 1, where the start error is the
sup norm over 1 024 x 192 potentials - the worst row, which a mean-squared regression does not bring down much. (The
two runs' systems differ after the first steps, so the per-report sweeps of the two columns are of different weights;
ms a step is the shared machine's, load 5-19.) What would make it pay - regressing in the sup norm or on the residual,
predicting from the token as well as the stream, or a warm start from the previous step's state at the same position -
is not tried here.
