# oann - hand-off note

State at the cut-over (C3, 2026-11-15): the redesign of docs/redesign.md is built and the old modules are gone.
Branch `pre-c3` is the last commit with them (`git show pre-c3:linalg.olang`).

## Where things are

- Layers: `tensor`, `kernels/`, `ad/`, `dyn/`, `nn/`, `models/`, `data/`, `io/`, `board/`, `tools/` (README.md).
- `make test OLANG=path/to/olang` runs every module's tests; the programs are the makefile's other targets.
- olang: a current master compiler (text joins `+`, the null test `== null`, a catch's `default` taking the whole
  expression). No migration step is outstanding; `olang fix .` has nothing left to do here.

## Removed at C3

`linalg kernels ops nn layers conv vision circuit board sparse optim train generate quant checkpoint` (root modules,
replaced as docs/redesign.md section 2 says), `datasets/` (replaced by `data/`), and bench drivers and repro cases that
imported them. Every old test is covered by a test of the new code or was a parity check that went with its module
(docs/redesign.md, "As built (C3)" under 4.2).

Moved to oagent (`agent/`): `agent`, `store`, `consolidator`, their examples (`sleep_retention`, `nights`,
`bandit_settle`), `bench/paired`, and the frozen engine they run on (`circuit`, `linalg`, `optim`, `sparse`, `nn`,
`ops`, `conv`, `kernels`, `quant`).

## The agent's mechanisms as dyn pieces (wt-dynpieces, 2026-11-18)

Both were already in dyn: the top-k store as a connection (`dyn/store`'s `Memory`, the `Remember` rule, over
kernels/sparse) and sleep consolidation as a rule (`dyn/sleep`'s `Sleeper`, the `Dream` rule); oagent's legacy engine is
gone (oagent 5ae86cd). What oagent's consolidator offers and the Sleeper lacked is added: `Parts` (slow answer, store
read, familiarity apart), a checkpoint's state (`Generator`/`SetGenerator`, `CueInputs`, `Keep`), and dawn rewriting at
rate 1 whatever the store's day Rate. Tests: Memory's recall against a dense connection masked to the top K (1e-12), a
restored sleeper's night bit for bit. The agent's slow part stays a gated recurrence on ad/graph (BPTT); rebuilding it
as a dyn system on the Sleeper is oagent's step.

## The agent's slow part on dyn (wt-dynagent, 2026-11-18)

The two pieces oagent's consolidator had and the Sleeper lacked: `dyn/recur`'s `Recurrence<T>`, the consolidator's
gated linear recurrence on ad/graph trained by BPTT (its answers and gradients checked against the recurrence written
by hand and its central differences, 1e-12 and 1e-5 relative), as the Sleeper's slow part (`Sleeper(l, m, ..., seed,
rec)`: the system supplies the key/value populations and the store; dreams are the recurrence's answer plus the
store's read; nights teach it by BPTT and AdamW); and `DayRate` (off by default; the old path is unchanged): the
consolidator's day step, a backtracking Armijo line search on the slow part's loss against the targets less the
store's read (its own answer where unobserved) - matched against that update written by hand on a toy case (1e-8),
and on a system's slow part through a centered EP lesson. A sleeper's key is the whole row, so a recurrence of Window
1 (the agent's) is the consolidator's; a longer window keys the store by the whole sequence, not step by step.

## Small graphs (bench/smallgraph, 2026-11-16)

oagent's consolidator moved from its own engine to ad/graph and its nights got 33-40% slower. bench/smallgraph is its
shape: BPTT over 12 steps of a gated linear recurrence, 11 inputs, 16 hidden, batch 16, F64, AdamW; one round a Forward,
Backward and step. Best of 3 x 3000 rounds, runs alternated on a loaded machine:

| | ns a round |
|---|---|
| master e2ac6c0 | 294-310k |
| now | 121-125k |
| the old engine (oagent 9894854, `agent/legacy/nn.olang`, the same graph) | 86-99k |

What it cost, and what changed:
- **Views made at every read** (a quarter of the step): `Value`/`Grad` built a tensor (shapes, strides, a check) at each
  call, several per node. A captured graph now makes every node's value and gradient view once per batch size in use
  (`makeViews`, at a Forward on a new n) and reads them from two arrays; an eager graph still makes them as it reads.
- **Sigmoid, Tanh and Relu's backward from their output** (`elem.ActGradOut`): y (1 - y), 1 - y^2, y > 0, instead of
  the exponential again from x. The plan keeps the output to the backward and lets the input go (`readsOwn`,
  `readsInputs`); the transformer's arena is unchanged (45.2 MB, bench/arena_ad).
- **The activations called the exponential per element** in a large program (12 ns an element against 2): olang links a
  program over 8 MB of IR by ThinLTO, which does not inline fastExp64 into an instance made in another object
  (repro/crossinline.olang). Sigmoid and Tanh over F64/F32 now run through array-wide forms compiled beside it
  (`elem.SigmoidInPlace64` and the rest).
- **Left: the products.** The same gemm call (16 x 16 x 11) takes 614 ns in a small program and 1379-1456 ns in one
  importing ad/graph, for the same reason (repro/crossinline.olang) - most of the gap to the old engine, whose bench is
  a small program linked by full LTO. A fix belongs in olang's link (ThinFrom, ImportLimit, vectorizing at a ThinLTO link),
  not here. bench/arena_ad's transformer step is unchanged (395-447 ms before, 410-417 after).

## Kernel follow-ups (wt-oannkern2, 2026-11-16)

Measured on the shared 4-core Zen 4 (load 4-13), medians of 5-7 interleaved rounds, olang-bin of 2026-11-16:

| | before | after |
|---|---|---|
| BF16 weights, batch 1 (`bench/storage`): 1 x 784 x 128 / 1 x 1024 x 1024 against F32 | 0.19x / 0.63x | 0.62-0.68x / 1.9-2.0x |
| `bench/int8`'s F32 baseline, 64-1024 rows | 12-17 GFLOPS | 80-142 GFLOPS |
| INT8 against F32 at 64-1024 rows (`bench/int8`, dynamic / static) | 1.04-1.46x (against the slow baseline) | 0.53-0.71x / 0.58-0.75x |
| INT8 at batch 1 (`bench/int8`, static): 1 x 128 x 512, 1 x 512 x 128, 1 x 128 x 128, 1 x 1024 x 1024 | 0.97, 0.94, 0.99, 3.29x | 1.39, 1.03, 1.36, 3.56x |

- BF16: `Gemv`'s two rows against one read of the vector made LLVM pair the rows' widenings in 128-bit pieces; BF16
  rows now go one at a time in two 16-lane vectors (`vpmovzxwd`, `vpslld`, `vfmadd231ps` on whole registers). At L2-resident
  sizes the widening's two shuffle-port instructions per 16 values keep it below F32; where the weights stream from
  memory it reads half the bytes and is 2x. F32 rows are summed in the same order (two vectors a row), so BF16 storage stays
  its widened product bit for bit; INT8 weights widened through that path went 0.92x -> 0.78-0.90x at 1 x 784 x 128
  (GemmI8 is INT8's own path). `vdpbf16ps` would need the activations in BF16 too, and olang does not form it.
- BF16 matrix-vector, looked at again (wt-bf16mv, 2026-11-18; Sapphire Rapids 4-core VM, load 15-24, medians of 9-15
  interleaved rounds): batch 1 against F32 is 0.49-0.66x at 1 x 784 x 128, 0.72x at 1 x 128 x 512, 0.51-0.63x at
  1 x 512 x 128 and 1 x 512 x 512 (weights in L2), 1.42-1.73x at 1 x 1024 x 1024 and 1 x 4096 x 1024 (streaming). Why:
  `dotNarrow`'s row has two FMA chains (25 steps x 4 cycles a row at depth 784, plus a 15-add `laneSum`), where F32's
  `dot2` has four over two rows; the scheduler fills with each row's waiting FMAs, so rows barely overlap. Every form
  with more than two chains in one loop is taken apart by LLVM 18's SLP vectorizer, which pairs the chains' BF16
  widenings vertically (`vmovw`/`vpinsrw` gathers, 0.07-0.21x): two rows (as before), two rows against two copies of
  the vector, the second row's lanes reversed or `+ 0.0`, four vectors or a 32-lane accumulator for one row, one row
  widened into a buffer and the other in place, the widening written as `F32.FromBits(U32(b.Bits()) << 16)` (worse:
  `Bits()` is kept opaque per element). What does vectorize - both rows widened into an L1 buffer first, then F32's
  `dot2` (0.49-0.63x, streaming down to 1.06-1.32x), or rows cut into 256-value blocks each summed apart (0.52-0.63x,
  F32 itself slower) - is no faster. Left as it is; the fix belongs in olang's codegen (a BF16 widening SLP keeps
  horizontal, or explicit lanes), and `vdpbf16ps` (avx512_bf16 is on this CPU) stays out, needing BF16 activations.
- BF16 matrix-vector, the cause found (wt-bf16vec, 2026-11-18; olang O45): not olang's widening IR but LLVM's SLP
  order - it visits the loop's exit before its header, so a total of two chains' own `laneSum`s (or two rows' results
  stored side by side) seeds a 2-wide tree pairing the chains' lanes back into the loop; for F32 that tree costs more
  than it saves, for BF16 the widening tips it. `dotRow` now sums every row, any storage type, in eight chains added
  lane by lane before one `laneSum` (the order is shared, so the bit-for-bit storage tests hold), one row at a time:
  a clean zmm loop (vpmovzxwd, vpslld, FMA). Batch 1, 9 rounds alternated with the old build three times (load 10-17):
  BF16/F32 1 x 784 x 128 0.44-0.60x -> 0.69-0.75x, 1 x 512 x 512 0.58-0.62x -> 0.74-0.86x, 1 x 128 x 512 0.67-0.73x ->
  0.80-0.84x, 1 x 1024 x 1024 1.42-1.70x -> 1.46-1.70x; F32 unchanged within noise; INT8 weights under F32 unchanged at
  cache-resident shapes, 2.1-3.2x -> 1.7-2.6x at 1 x 1024 x 1024 (it lost the two-row read of the vector). BF16 >= F32
  stays out of reach in L2: a widened vector is three vector-port uops to F32's one.
- `bench/int8`'s F32 ran slow because nn/quant imported ad/graph: the program passed ThinFrom and linked by ThinLTO,
  which inlines no kernel helper across objects (repro/crossinline). nn/quant is now graph-free (`DenseOf`, `ConvOf`)
  and nn/quantgraph reads models from a graph. Against the real F32 path INT8 is slower at 64+ rows: its packed
  micro-kernel (`vpdpwssd`, 16 products an instruction, widened from bytes) runs at 50-97 GOPS against F32's 80-146.
  Batch-1 products still win where the weights stream from memory (3.3-3.6x at 1 x 1024 x 1024).
- One-row INT8 on olang's `a.Sum<A>()` (2026-11-16): `dotI8` and the four-row `dot4I8` are `(x * w).Sum<I32>()` over the
  U8 and I8 rows, which builds to `vpdpbusd` (repro/vnnidot removed). From depth 256 (`i8SumDepth`); shallower, four
  plain sums against one read of the row stay (1 x 128 x 128: 1.1 us plain, 1.9 us Sum). bench/int8 batch 1, old -> new,
  two alternated runs: 1 x 784 x 128 5.0-5.9 -> 2.6-3.2 us (1.8-2.0x); 1 x 512 x 128 3.3-3.6 -> 1.5-2.9 us (1.2-2.4x);
  1 x 1024 x 1024 54-81 -> 24-27 us (2.2-3.3x; 2.1-2.9x -> 5.0-6.3x of F32); 1 x 128 x 512 and 1 x 128 x 128 unchanged
  (they take the plain path). The packed micro-kernel (an outer product, one broadcast byte times a vector of weights) has no
  four-adjacent-products shape and is unchanged; `vdpbf16ps` stays unreachable (olang does not use it: it reorders a float sum).
- Padding in GemmI8's dot products (`dots`, `dotsPay`; 2026-11-17, olang 5f85a388's masked-tail LaneDot, two alternated runs of
  bench/int8 and bench/int8cnn, another build running): kept. Unpadded (depth k as is, `dotsPay` from k >= 64) made 64 x 784 x
  128 static 104 -> 126 us and 128 x 784 x 128 217 -> 245 us (slower), 25088 x 144 x 32 4.8 -> 5.8 ms, the CNN at batch 64-128
  1.4-1.5x -> 1.2-1.3x of F32; only 128 x 128 x 10 gained (10.7 -> 8.5 us). Dropping the dot path (everything to the
  outer-product tiles) was 2-3x slower at depth 784 (64 x 784 x 128 74-103 -> 214-221 us). The masked tail is not as fast as
  whole 256-blocks over a long depth.
- Next here: the packed INT8 micro-kernel - done 2026-11-17 on olang's grouped Sum, see "INT8 on the grouped lane total" below;
  ad/ops' `Scan` is `GatedScan` (the prelude now has a `Scan`).

## Pruning at run time (wt-oannmeasure2, 2026-11-16)

docs/redesign.md question 3 beyond the masks. `Rule.Prune(by, target, pace, regrow)` (dyn/learn's `Prune`, on any
clock) thins a module's learnt dense, symmetric, sparse-row and event-driven connections as one pool: each live weight
is scored (`conn.Score`: `Magnitude` |w|, `Activity` |w| mean|s_pre|, `Energy` |w mean(s_post s_pre)|, its term of the
bilinear energy) over its connection's mean, and the lowest `pace` of the pool are masked a tick, down to `target`;
`regrow` revives that fraction of the live count among masked weights whose units coincide most (and masks as many more);
a hidden unit left with no live weight is retired. All of it is pinned allocation-free (`Learner<F32>.Prune`, `Hold`,
`Revive`, `Ends`) and runs on the Step clock in the resident-memory test. `System.Sparsify(below)` is the one step that
allocates: each dense block with at most `below` of its weights live becomes compressed sparse rows over its own region
(read both ways if it was symmetric), the same equilibrium to 1e-9 (test), its products costing its live weights;
an optimizer made before is made again. `System.Hold()` after an optimizer step puts masked weights its momentum moved
back to zero. `System.Macs()` counts a settle's multiply-adds as the structure stands (once: folded inputs; each sweep:
recurrent carries, twice where reciprocal).

Ranking raw scores across connections emptied the readout first (its weights are kept small by `Restrain`) and the
accuracy fell to chance at 15% kept; scores over their connection's mean keep every path.

Result (`make prune ARGS="energy 0.05"`, examples/prune_settle.olang): XOR of the signs of two coordinates beside 2
noise inputs, 256 training and 256 unseen points, 4 inputs -> 32 tanh hidden (lateral, symmetric) <-> 2 readouts, EP
(beta 0.5) and AdamW 0.05, gain restrained to 0.9; 2000 lessons, pruned 5% of the weights every 50 lessons, 2000 lessons
again at the end, then `Sparsify(0.5)`. Accuracy on the unseen points and multiply-adds of one point's settle:

| seed 1, energy score | weights kept (x->h / h<->h / h<->y) | test | a sweep | a settle (sweeps) |
|---|---|---|---|---|
| dense | 1216 (128 / 1024 / 64) | 86.3% | 2176 | 13184 (6) |
| masked 40% | 484 (54 / 376 / 54) | 87.1% | 2176 | 8832 (4) |
| masked 20% | 240 (49 / 147 / 44) | 85.9% | 2176 | 17536 (8) |
| masked 10% | 118 (31 / 69 / 18) | 84.0% | 2176 | 17536 (8) |
| 5%, relearned, sparse rows | 61 (25 / 26 / 10) | 80.5% | 72 | 529 (7) |

Ending at each fraction (relearned, sparse rows), test accuracy dense -> pruned and multiply-adds a sweep / a settle:

| seed | dense | 20% kept | 10% kept | 5% kept |
|---|---|---|---|---|
| 1 | 86.3%, 2176 / 13184 | 85.9%, 430 / 3489 | 84.4%, 178 / 1279 | 80.5%, 72 / 529 |
| 2 | 82.8%, 2176 / 15360 | 83.2%, 394 / 2411 | 47.7%, 194 / 1189 | 64.1%, 80 / 661 |
| 3 | 62.9%, 2176 / 11008 | 80.5%, 464 / 6536 | 51.2%, 184 / 1134 | 60.5%, 62 / 712 |

- 20% kept holds the dense accuracy on every seed (seed 3's dense run had not learned; pruning the noise inputs and the
  lateral block let it); 10% holds on seed 1 and falls apart on two of three at pace 0.05. Pace 0.02 at 10%: 83.6 /
  79.3 / 57.0%; with regrowth 0.01 a tick: 73.4 / 85.5 / 60.5% - neither is better on every seed.
- Scores at 5% (seed 1): energy 80.5%, activity 65.6%, magnitude 63.3%.
- A sweep costs the same until `Sparsify` (masks are multiplied); after it a sweep is 5-30x cheaper. Sweeps a settle vary
  with the weights (4-11 here), so the per-sweep count is the like-for-like one.
- `Sparsify` keeps an optimizer's moments only if the caller remakes it. Tiles, ranks, the schedule and the skip
  connection: the next section.

## Pruning: schedule, tiles and ranks, the skip connection (wt-oannprune2, 2026-11-16)

**The skip connection (x -> y, folded) is not a bug; it is what the certificate bounds.** The EP contrast through a
folded input into a readout is the cost's gradient (test: `Learner.Lesson` "a skip connection from the inputs into the
readout", relative error < 1e-5 against central differences). The failure is the scale: `Certify`/`Restrain` cap every
unit's incoming recurrent mass at the gain (0.9), so the hidden layer's whole vote into a readout unit is at most 0.9
in its net input, while a folded connection is outside the certificate (it carries a constant) and is not capped. The
skip is drawn Glorot-uniform for a 4 -> 2 block (bound 1) over inputs in [-3, 3]: its carry reaches several units, a
linear function of the inputs, wrong on half of the points for XOR, and no hidden unit can outvote it (test:
`System.Restrain` "the gain caps ... not a folded skip"). Started from zero and learnt freely it grows the same way
(to |w| 1-2 in 2000 lessons: the linear fit of the training sample's noise) and the readout saturates (seed 1: 55.9%). Without
`Restrain` nothing learns (the settles stop converging, 1998 of 2000 lessons refused, skip or not). Held to an incoming
mass of 0.1 a readout unit and started from zero, the skip is harmless (`make prune ARGS="... cubic zero"`):

| seed | no skip | skip drawn | skip from zero, mass held to 0.1 |
|---|---|---|---|
| 1 | 86.3% | 51.6% | 89.1% |
| 2 | 82.8% | 50.8% | 84.8% |
| 3 | 62.9% | 64.1% | 70.7% |

So a skip into a certified system's readout has to be scaled to what the recurrent path can carry; nothing in oann
does that for a folded connection (a rule could: a gain-like cap on a readout's folded mass), and none is built.

**Cubic schedule (Zhu and Gupta 2017) is the default.** `conn.Schedule { Linear, Cubic }`, a field of `Rule.Prune`
and a parameter of `Learner.Prune` (default Cubic): the fraction kept after tick t of n = (1 - target) / pace is
target + (1 - target)(1 - t/n)^3, the tick read off the fraction now kept (so a rule stays stateless and regrowth or
outside masking does not upset it); the same number of ticks as Linear (test `paced`). XOR beside noise (as above,
energy score, pace 0.05, relearned, sparse rows), test accuracy:

| kept | schedule | seed 1 | seed 2 | seed 3 |
|---|---|---|---|---|
| 10% | linear | 84.4% | 47.7% | 51.2% |
| 10% | cubic | 83.6% | 82.8% | 80.1% |
| 20% | linear | 85.9% | 83.2% | 80.5% |
| 20% | cubic | 84.8% | 84.4% | 80.5% |

At 20% the two tie (83.2% mean each); at 10% the linear pace's collapses on seeds 2 and 3 are gone. Sweeps a settle
rise at 10% cubic (8-17 against 6-7): a sweep costs 164-182 multiply-adds, a settle 1412-3125.

**Tiles by tile, LowRank by rank.** Both are prunable now: each tile or rank is one item of the pool, scored as one
(a tile: the sum of its weights' scores; a rank a b^T: |a| |b|, |a| sum |b_j| mean|s_j|, or |a^T mean(s_post s_pre^T)
b|, read off the coincidence's gradient of the first factor), taken over its connection's mean and counted as the
weights it holds (`Link.Full`: a tiled one's stored tiles, a low-rank one's factors; `Group`). A masked tile is dead
in the kernels at once; a masked rank's zeros are still multiplied until `Compact`, which moves the live ranks down in
place and makes the rank smaller (`Macs` follows; allocation-free; an optimizer made before is made again). Regrowth
stays with the one-weight connections. Test: `Learner.Prune` "a tiled connection loses whole tiles and a low-rank one
whole ranks".

**Compacting retired units out of the populations (a re-plan): measured 2026-11-17, not built - it cannot pay on MNIST.** What it
would need: a `System.Shrunk()` that records a new system from the old one's records (modules, populations at their live sizes,
signals, rules) and re-plans it; every connection rewired by the map of live units - dense, symmetric and event-driven blocks
as their live rows and columns, a low-rank one as A's live rows and B's live columns, sparse rows with their old pattern
restricted (Link has no way to take a pattern from outside `Draw` yet), tiles refused or turned dense; biases, kept state
fields, drives, targets and labels copied by the same map; the learner and optimizer made again. What it would win, measured
(`make mnistlearn ARGS="2 60000 128 32 1 0.001 0.5 1 0.2 0.1 200 1"`, 128 hidden units, 20% of the weights kept, accuracy
97.43%, `silent` in mnist_learn.olang counts the units before `Sparsify`): of 128 hidden units 7 (5.5%) have no live weight
out, none is without every weight. Magnitude pruning thins each unit's rows evenly; it does not empty units. A unit nothing
hears costs its pixel row (about 156 of the 20013 multiply-adds once a settle, 0.8% each, at most 5% for the 7) and its share
of a sweep's elementwise work; a sweep's 628 multiply-adds are live weights only (sparse rows), which compaction does not
change. Under 5% of a settle at the most, against ~10% needed and a re-plan of the whole system; the retired units' weights are
better dropped by the prune itself (a unit nobody hears loses its incoming weights) than by shrinking populations. Upper bound timed (wt-oanncompact, 2026-11-17): the same
run with 121 hidden units (the 7 silent ones never made; `make mnistlearn ARGS="1 6000 H 32 1 0.001 0.5 1 0.2 0.1 50 1"`, H = 128
then 121, four alternated pairs, sparse-rows settle of a batch): 140.6 / 138.9 / 143.9 us at 128 against 136.5 / 137.7 / 140.4 us at 121
(one pair noisy), i.e. ~2%; multiply-adds a settle 26627 -> 25614 (-3.8%); accuracy equal. Not merged: Shrunk was not built, it cannot pay
~10%. Revisit if a
pruning schedule that retires whole units (a group score per unit) is built.

**MNIST (`mnist_learn`) pruning:** measured 2026-11-16, see "Measured on the data" below.

## INT8 packed products on vpdpbusd (wt-oannvnni, 2026-11-16)

`GemmI8`/`GemmI8Left` from 4 rows take padded dot products (`dots` in kernels/gemm) where they pay: each activation
row and weight row is copied once with its depth padded by zeros to a multiple of 256, and each output is one
`(x * w).Sum<I32>()` of whole blocks - `vpdpbusd`, 64 products an instruction, checked in the binary's inner loop
(4 `vpdpbusd` a block, then a horizontal sum). Taken from depth 192 padded by at most half, or from 128 padded by at
most double where the outputs fit the thin tile (n <= 16); elsewhere the outer-product tiles stay. Integer sums, so
bit for bit the old results (the tests, the MLP's and CNN's accuracies unchanged). The VNNI layout the brief asked
for - b as [k/4][16][4], a's four bytes broadcast - does not build to `vpdpbusd` in olang: four adjacent products a
lane summed into I32 becomes `vpmaddwd` on widened bytes; LLVM forms `vpdpbusd` only from a reduction to one value
(repro/vnnitile); the 16-bit pair packing (a [k/2][32][2] I16 panel) became `vpmulld`, worse. That lowering, or LaneDot keeping a vector accumulator across its blocks, is what would close the rest.

Measured one core, AVX-512 VNNI, alternated before/after runs on the shared machine (static INT8 against F32):

| | before | after |
|---|---|---|
| bench/int8 64 x 784 x 128 / 128 x 784 x 128 | 0.58x / 0.60x (48-55 GOPS) | 1.27-1.33x / 1.08-1.16x (92-111 GOPS) |
| bench/int8 512^3 / 1024^3 | 0.64x / 0.72x (72-79 GOPS) | 0.99-1.23x / 1.21-1.23x (121-151 GOPS) |
| bench/int8 128 x 128 x 10 (thin, k 128) | 0.23x | 0.97x |
| bench/int8 1024 x 512 x 128 / 1024 x 128 x 512 (k 128, wide: old path) | 0.61x / 0.59x | unchanged |
| bench/int8 convolutions 100352 x 9 x 16 / 25088 x 144 x 32 | 0.15x / 0.40x | unchanged (too shallow to pad) |
| MLP (`make int8 ARGS="mlp 3"`) batch 1 / 64 / 128 | 2.6x / 0.62-0.71x / 0.55-0.57x | 2.6x / 1.14-1.31x / 0.96-1.10x |
| CNN (`ARGS="cnn 1"`) batch 1 / 64 / 128 | 0.79x / 0.75x / 0.84x | 0.69x / 0.88x / 0.84x (noise; its convolutions are depth 9 and 144) |
| bench/int8cnn batch 1 / 64 / 128, two runs each | 0.68x / 1.01-1.02x / 1.01x | 0.67-0.81x / 1.04-1.09x / 1.04-1.09x |

Batch 1 takes the one-row path, unchanged. The CNN stays at F32's speed or below: its products are 9 and 144 deep
(100352 x 9 x 16 runs at 4 GOPS, 7x F32's time - not yet profiled), and the rest of its forward is not INT8.

## INT8 on the grouped lane total (wt-int8dots, 2026-11-17)

`GemmI8`/`GemmI8Left` from 4 rows now take `grouped` (kernels/gemm): b packed as [k/4][nr][4] bytes (depth padded to 4 only),
each row of a copied once, and a tile step is `acc += (a's four bytes * the panel's 64 bytes).Sum<I32>(4)` (olang E39) =
`vpbroadcastd` + `vpdpbusd`, 6 x 32 (wide) or 12 x 16 (thin) tiles. No horizontal sum, no padding to 256: the dot-product
path (`dots`, `dotsPay`, `packRowI8`) is removed. Depth up to 32 (`i8Shallow`) keeps the outer-product tiles with their
shallow micro-kernel (depth 9: grouped pads 9 to 12 and measured no faster). Integer sums, bit for bit the same (the gemm
test's reference, sizes added at depths 9, 144, 784, 785 and odd shapes). Batch 1 untouched.
One olang oddity: accumulating `t.v[a:b] += (x * b).Sum<I32>(4)` straight into a `mut` parameter's array made
`GemmI8.Allocates` true; into a local accumulator copied out at the end (as the old tiles) it does not.

bench/int8 static INT8 microseconds, median (min) of 7 alternated runs, one core, the machine shared (F32 times swung 2x
between runs, so ratios to F32 are from the same-run pairs only where stated):

| shape | padded dots / old tiles | grouped | INT8 / F32 (grouped, same runs) |
|---|---|---|---|
| 64 x 784 x 128 | 89.9 (78.8) | 70.2 (46.1) | 2.3x |
| 128 x 784 x 128 | 173.9 (153.7) | 120.3 (77.2) | 2.3x |
| 25088 x 144 x 32 (depth 144) | 5518 (3285) | 2451 (1600) | 4.3x |
| 100352 x 9 x 16 (depth 9) | unchanged path (noise 1370-4025 vs 2160-3080) | | 0.4x |
| 128 x 128 x 10 | 14.2 (5.9) | 6.9 (4.7) | 1.3x |
| 1024 x 128 x 512 / 1024 x 512 x 128 | 3784 / 1043 | 683 / 560 | 1.9x / 2.6x |
| 512^3 / 1024^3 | 3181 / 38268 | 1060 / 11791 | 4.1x / 5.6x |

bench/int8cnn static at batch 64 / 128 (3 alternated runs): 3.1-3.8 / 9.0-13.9 ms -> 2.1-3.9 / 4.5-9.4 ms (F32 11-19 ms). Depth 9 (the first
convolution) stays 0.4x of F32 alone: it is the shallow tile, bound by its quantize and dequantize passes.

## INT8 convolution at depth 9 (wt-oannconv, 2026-11-16)

Profile (`bench/convprof.olang`, 100352 x 9 x 16, one core, per call): quantize 0.2 ms, `GemmI8` 5.1-5.9 ms, bias and ReLU 0.34-0.44 ms.
The product was nearly all of it: at depth 9 the thin tile's sixteen-wide outer-product steps run for 9 steps and the pack of the
activations (`packI8`) read each row with a stride per panel step. Two changes in kernels/gemm.olang: `packI8` walks the panel step by
step (the writes sequential, the reads one per row), and a depth up to 32 takes `microI8Shallow` - each tile row its own sum over the
depth, one vector of outputs against the weights' panel. Bit-exact (the gemm tests against the reference pass). `GemmI8` 5.1-5.9 -> 2.1-2.5 ms
(2.4x); the rest (quantize + dequantize) is now 25% of the call and not fused: it is memory-bound at one pass each.
Alternated old/new runs, F32 / INT8 (static) in `bench/int8cnn`:

| batch | before dynamic / static | after dynamic / static |
|---|---|---|
| 1 | 0.72x / 0.72x | 1.00x / 0.99x |
| 64 | 0.89x / 1.08x | 1.31x / 1.63x |
| 128 | 1.01x / 1.00x | 1.33x / 1.45x |

`bench/int8` 100352 x 9 x 16 still reads 0.34x / 0.44x of F32 alone (F32's depth-9 product runs at 22 GFLOPS, INT8's 7-10 GOPS with its
quantize and dequantize passes), but the whole CNN is now ahead of F32 at batches 64 and 128. Not done: fusing im2col with quantize
(the patches are built in F32 then quantized; one pass saved of three) - a direct kernel would save under 0.2 ms of 12 ms. Accuracy is untouched (the
products are bit-identical). The batch-1 path is the old one-row dot products.

## Settle options: Anderson, Gauss-Seidel, Contraction (wt-oannsettle, 2026-11-16)

docs/redesign.md's three unbuilt settle features, in dyn: `settle.Sweeping(Anderson, GaussSeidel)` in `Budget.Scheme`, set for a
system by `System.Use` (its `BudgetFor` and a Learner's budgets carry it; Anderson's history is reserved there, so a settle still
allocates nothing), and `System.Contraction()`. Tests: the same fixed point as plain sweeps (every wiring, free and nudged; 2e-10
without the event-driven connection, whose 1e-5 threshold makes the end path-dependent by ~1e-6 with it), fewer sweeps, bit for bit
in tasks; a one-way chain made in reverse order settles in one Gauss-Seidel sweep (Jacobi: 3); EP's centered contrast still the
cost's gradient against central differences (1e-5) with each; `Contraction` exactly kappa on the uniform block, the dense
coupling's largest singular value on `small`, below the gain on `mixed`.

Measured on a loaded machine (load 5-16 on 4 cores; logs/ in the worktree, not committed): `mnist_learn` 784-128-10, batch 32, seeds
1 / 2, dense 3 epochs (`3 60000 128 32 SEED 0.001 0.5 1 1 0.1 200 1 0 WAY`) and pruned to 20% (`2 60000 128 32 SEED 0.001 0.5 1 0.2
0.1 200 1 0 WAY`, after Sparsify), and `xor_settle` (300 lessons, best of 3 runs). Sweeps a lesson (free + nudged, epoch 3 or the
relearn epoch), sweeps an answer (a test-set free settle), wall time of a test settle (a batch of 32), refused lessons, accuracy:

| MNIST dense | sweeps a lesson | sweeps an answer | us a settle | epoch s | refused | accuracy |
|---|---|---|---|---|---|---|
| plain (Jacobi) | 32.4 / 32.0 | 14.7 / 14.2 | 258 / 386 | 2.6 / 2.6 | 0 / 0 | 96.71 / 97.10% |
| Anderson 5 | 29.4 / 31.2 | 14.7 / 16.4 | 511 / 589 | 3.9 / 3.3 | 0 / 1 | 96.71 / 96.85% |
| Anderson 3 | 29.7 / 31.2 | 14.9 / 15.8 | 498 / 507 | 3.6 / 3.6 | 0 / 0 | 96.65 / 97.05% |
| Gauss-Seidel | 21.1 / 21.7 | 7.8 / 7.4 | 217 / 191 | 2.2 / 2.2 | 0 / 0 | 96.63 / 97.09% |
| both (GS, Anderson 5) | 16.6 / 16.3 | 5.8 / 5.7 | 233 / 232 | 2.6 / 2.8 | 1 / 1 | 96.69 / 96.96% |

| MNIST pruned 20% | sweeps a lesson | sweeps an answer | us a settle | refused (relearn) | accuracy |
|---|---|---|---|---|---|
| plain (Jacobi) | 42.7 / 43.4 | 18.7 / 19.9 | 236 / 296 | 0 / 0 | 97.43 / 97.52% |
| Anderson 5 | 41.5 / 44.5 | 20.3 / 24.3 | 684 / 460 | 6 / 12 | 97.38 / 97.36% |
| Anderson 3 | 42.1 / 46.2 | 20.5 / 24.3 | 635 / 598 | 6 / 14 | 97.33 / 97.41% |
| Gauss-Seidel | 27.8 / 28.4 | 10.2 / 10.8 | 229 / 223 | 0 / 0 | 97.30 / 97.40% |
| both | 21.4 / 21.6 | 6.7 / 6.9 | 359 / 362 | 0 / 0 | 97.37 / 97.37% |

| XOR (8 hidden) | sweeps a lesson | us a lesson | loss at 300, right |
|---|---|---|---|
| plain | 9.1 | 7.7 | 0.0001, 4/4 |
| Anderson 5 | 6.7 (1 refused) | 7.6 | 0.0005, 4/4 |
| Anderson 3 | 7.1 | 7.3 | 0.0006, 4/4 |
| Gauss-Seidel | 8.0 | 7.6 | 0.0005, 4/4 |
| both | 5.8 | 6.6 | 0.0005, 4/4 |

Reading: Gauss-Seidel saves a third of a lesson's sweeps and half an answer's on MNIST (the hidden population is swept before the
readout, which then reads it new) and is as fast or faster in wall time everywhere, but it is level on XOR's time and its accuracy is
not unchanged (-0.01 to -0.13 points, lower in all four MNIST runs; the trained path moves with the fixed point's error, within the
tolerance). Anderson cuts sweeps where the system stays near certified (XOR, MNIST's first epochs: 20 against 30 a lesson before the
safeguard was made strict) but costs more than a sweep here: MNIST's recurrent block is 128 x 10 (2560 multiply-adds a row), so its
gather, Gram updates and extrapolation over 138 units a row are as dear as the sweep they replace, and on the trained, uncertified
system (spectral bound past 1, below) it stalls and some lessons are refused. Its first safeguard (only dropping the history where the residual grew)
saved more sweeps (20-24 a lesson) but refused 4-12 lessons an epoch; the strict one (back to the plain step from the last accepted
state) refuses fewer and saves little. **The default stays Jacobi**: no option is faster on every case with accuracy unchanged.
Anderson should pay where a sweep is dear (wide recurrent blocks) and the system is certified; Gauss-Seidel is the option to take
where sweeps count (an answer's latency). `Contraction()` (printed now by both programs) reads kappa where the sup-norm gain over-counts: MNIST certified at gain 0.900 has
a spectral bound of 0.107 (the 784-wide input is folded; the 128 x 10 symmetric block's singular value is small), and after
epochs 1-3 the gain is 7.7 / 8.9 / 10.1 against a spectral bound of 0.97 / 1.18 / 1.37 - the trained system leaves even the
Euclidean certificate from epoch 2, where Anderson stalls. XOR at lesson 300: gain 6.13, spectral bound 3.12.

## The certificate's norm: sup or spectral (wt-oanncert, 2026-11-17)

`System.Bound` (`Norm.Sup`, the default, or `Norm.Spectral`) picks the norm `Certify`, `Restrain`, `Rate`,
`LessonTolerance` and `BudgetFor` hold kappa in (`Kappa()`); the spectral bound by `Engine.Spectral`, now warm-started
from the vector the engine keeps (laid out by `Ready`, so a lesson's `Restrain` allocates nothing). What changes against
settling.md 1.2 is in docs/redesign.md 1.4 (the distance bound loses `sqrt(n)` per unit under a unit-by-unit residual
test, so `LessonTolerance` divides by it; the cross-entropy nudge's off-diagonal terms are uncovered in either norm;
EP's theorem and the refusal test need no norm; the board stays on the sup norm). Tests: the spectral bound against a
dense SVD (cyclic Jacobi on W^T W) on four random systems, certified and restrained to the target, `Kappa` following a
step; a system certified spectrally with its gain above 1 converges within its budget, cold and warm to one fixed point.

Measured on a loaded machine (load ~10 on 4 cores; logs/ in the worktree, not committed): `mnist_learn` dense, 3
epochs, Jacobi, seeds 1 / 2 (`3 60000 128 32 SEED 0.001 0.5 1 1 0.1 200 1 0 plain CERT`); none = certified at the sup
norm when made, then free (as before); sup / spectral = certified in that norm and `Restrain(0.9)` after every step.
Epoch 3; time a lesson is the epoch's wall time over its lessons, the power iterations included (four warm calls a
lesson, a few 128 x 10 products each - well under 1% of a lesson's multiply-adds; the spread is the machine's load):

| MNIST dense | sweeps a lesson | sweeps an answer | us a lesson | refused (epochs 1-3) | gain / spectral bound | accuracy |
|---|---|---|---|---|---|---|
| none | 32.4 / 32.0 | 14.7 / 14.2 | 1551 / 1146 | 0 / 0 | 10.1 / 1.37, 10.2 / 1.34 | 96.71 / 97.10% |
| sup | 13.1 / 13.0 | 3.9 / 3.7 | 1058 / 946 | 0 / 0 | 0.90 / 0.10 | 91.05 / 90.83% |
| spectral | 42.2 / 43.0 | 12.8 / 13.1 | 1548 / 1801 | 7 / 7 (5 in epoch 1) | 6.8 / 0.90, 7.0 / 0.90 | 96.97 / 96.70% |
| spectral at the sup tolerance | 28.7 / 29.0 | 9.5 / 9.7 | 1383 / 1327 | 0 / 0 | 6.9 / 0.90 | 96.86 / 96.93% |

| XOR (8 hidden, 300 lessons) | sweeps a lesson | us a lesson (3 runs) | refused | loss, right |
|---|---|---|---|---|
| none | 9.1 | 14.6 / 11.6 / 9.9 | 0 | 0.0001, 4/4 |
| sup | 6.4 | 9.5 / 8.3 / 8.6 | 0 | 0.0108, 4/4 |
| spectral | 10.2 | 18.0 / 15.0 / 12.9 | 0 | 0.0021, 4/4 |
| spectral at the sup tolerance | 6.7 | 11.6 | 0 | 0.0023, 4/4 |

Reading: the sup-norm certificate held through training is far too tight for MNIST - at gain 0.9 the spectral bound is
0.10, the system is nearly feed-forward and accuracy falls to 91% (cheap: 13 sweeps a lesson, 4 an answer). The
spectral certificate at 0.9 lets the gain sit at 7 and keeps the accuracy of the uncertified run, so the Euclidean
certificate is the one that fits a trained system. But kept rigorously - the qualifying test unit by unit, the
tolerance divided by `sqrt(n)` = 11.7 for MNIST's 138 relaxing units - its lessons cost more sweeps than the uncertified
run's (42-66 against 32) and some nudged phases are refused (residual 1e-4 to 3e-3 at the budget: the cross-entropy
force's coupling, which neither certificate covers). Run at the sup-norm tolerance (the "spectral at the sup tolerance"
rows: the per-unit distance bound not kept) it is better than the uncertified run on every count. **The default stays
`Norm.Sup`** (the brief's condition, nothing refused, fails); the step that would make the spectral certificate the
default is a qualifying test in its own norm - each row's Euclidean residual against `tau`, which keeps 1.4's bound
without the `sqrt(n)` - and a certificate of the nudged phase that counts the cross-entropy force's coupling.

## The stop test in the certificate's norm; the spectral certificate the default (wt-oanncert2, 2026-11-17)

A settle under the spectral certificate now stops on each row's Euclidean residual: `settle.Budget.Euclid` (set by
`BudgetFor` and a learner's budget when `Bound` is `Norm.Spectral`) replaces the unit-by-unit test with the largest
row's `|r|_2` over its relaxing units, read from the sweep's step `(u' - u) / dt` after each sweep (one pass over the
state, beside the products; retired units left out; a spiking or custom system keeps the unit test). The distance bound
`|v - v*|_2 <= tau / (1 - kappa)` is then rigorous in the norm kappa holds, and per unit as well (`|x|_inf <= |x|_2`),
so `LessonTolerance` lost its `sqrt(n)`. The nudged phases' rate (`Rate(beta)`, `NudgedKappa`) now counts the
cross-entropy force's coupling: `kappa + beta c` at +beta (the shortened step absorbs the self-slope), `kappa + beta
(f + c)` at -beta (`c = f = 1 / (4 T)` per unit of the readout's slope, `neuron.ForceCoupling`; argument in
docs/redesign.md 1.4 item 10); at MNIST's beta 0.5, T 0.25 neither phase is certified by it (1.4 and 1.9), so the
nudged budget is the clamped one (q 0.95: 261 sweeps), and the phases are measured to converge instead (below; test
`System.Rate` at beta 0.5). `System.Bound` defaults to `Norm.Spectral`; the board refuses a spectral system
(`LowerError`: its ranges need the sup norm) and its tests and `mnist_board` set `Norm.Sup`; `mnist_learn` and
`xor_settle`'s `none` and `sup` keep the sup norm, so their rows below are last round's runs. Tests: a settlement's
residual under `Euclid` lies between the largest unit's and `sqrt(n)` times it; a certified system's cold and warm
settles agree to `2 tau / (1 - kappa)` over a row; a cross-entropy system's nudged pair at beta 0.5 settles in budget.
Five tests of other things keep the sup norm explicitly: the Gauss-Seidel comparison (`System.Use`: under the spectral
certificate a Gauss-Seidel nudged settle took 85 sweeps against Jacobi's 82 - its rate is proven for the sup norm only),
`Restrain`'s per-unit caps, `Retire` (on the same `mixed` system), and two learner tests taught without `Restrain`
(`Learner.Prune`, the resident-memory test: spectrally certified at 0.9 and then trained free, a free phase ran out of
its 261 sweeps at residual 0.0027 - the sup certificate's wide margin is what kept them settling). `make test`: 714
passed.

Measured as last round (`3 60000 128 32 SEED 0.001 0.5 1 1 0.1 200 1 0 plain CERT`, epoch 3, seeds 1 / 2, load ~11 on
4 cores; logs/ in the worktree, not committed). none and sup were run again at this load for the times:

| MNIST dense | sweeps a lesson | sweeps an answer | us a lesson | refused (epochs 1-3) | accuracy |
|---|---|---|---|---|---|
| none | 32.4 / 32.0 | 14.7 / 14.2 | 1495 / 1514 | 0 / 0 | 96.71 / 97.10% |
| sup | 13.1 / 13.0 | 3.9 / 3.7 | 1120 / 914 | 0 / 0 | 91.05 / 90.83% |
| spectral, row-Euclidean stop (now) | 31.8 / 31.9 | 10.6 / 10.9 | 1798 / 1773 | 0 / 0 | 96.98 / 96.90% |
| spectral, unit stop at tau / sqrt(n) (last round) | 42.2 / 43.0 | 12.8 / 13.1 | 1548 / 1801 | 7 / 7 | 96.97 / 96.70% |
| spectral, unit stop at tau (last round, not rigorous) | 28.7 / 29.0 | 9.5 / 9.7 | 1383 / 1327 | 0 / 0 | 96.86 / 96.93% |

| XOR (8 hidden, 300 lessons) | sweeps a lesson | us a lesson (3 runs) | refused | loss, right |
|---|---|---|---|---|
| none | 9.1 | 17.1 / 12.1 / 16.7 | 0 | 0.0001, 4/4 |
| sup | 6.4 | 14.2 / 14.8 / 10.6 | 0 | 0.0108, 4/4 |
| spectral, row-Euclidean stop | 8.5 | 14.9 / 14.6 / 14.4 | 0 | 0.0022, 4/4 |

Reading: kept rigorous, the spectral certificate now refuses nothing, its lessons cost what the uncertified run's do
(31.8 against 32.4 sweeps; 3 more than the non-rigorous unit stop at tau, the row norm being the stricter test), its
answers 25% fewer sweeps (10.7 against 14.5), and its accuracy is the uncertified run's (96.94% mean against 96.91%) and
5.9 points above the sup certificate's. The time a lesson is 1.2x the uncertified run's at equal sweeps: the four warm
power iterations a lesson and `Restrain`, and the row-norm pass. XOR: lower loss than the sup certificate (0.0022
against 0.0108), 2 more sweeps a lesson. **`Norm.Spectral` is the default.**

## The nudged pair certified from the free state (wt-oanncert3, 2026-11-17)

`System.NudgedRate(beta, free)` replaces the worst-case coupling (`beta / (4 T)` per unit of slope) by the readouts'
force bounded over the ball the nudged settle can reach, row by row: the start's residual is at most the free
residual plus `beta |F(s0)|`; a sweep that contracts by q on the ball of radius r with that residual at most
`(1 - kappa_beta(r)) r` keeps the settle in it, and `kappa_beta(r)` counts the softmax's probabilities and the
readouts' tanh slopes bounded over the ball (`neuron.ForceOver`: sup rows/columns, or in the Euclidean norm also
`lambda_max(S) <= (1 - p_y) + max_{i != y} p_i` for the whole Jacobian and `|p|^2` for the coupling;
docs/redesign.md 1.4 item 10). `Learner.Lesson` gives the nudged pair the budget of that rate where it certifies the
batch (`BudgetAt`), the clamped one otherwise; `Taught.Rate` and `Taught.Rows` (rows certified on their own) report
it. Tests: a confident readout's pair at beta 0.5 certified in both norms where `Rate(0.5)` is 1, every sweep
shrinking the residual by at most the rate; soundness over 96 random settles (seeds, kappa 0.2/0.5/0.8, readout
bias 0-4, beta 0.1-1, both norms; 45 certified, 51 refused): every certified settle's measured contraction between
consecutive states is at most q (the largest measured ratio 0.90 q), and the test fails if the force term is dropped.

Measured `3 60000 128 32 1` (default `none`: sup norm, free) and the same with `plain spectral` (kappa 0.9 held),
master against the branch alternated, 3 runs each, load 16-25 on 6 cores (logs/ in the worktree):

| MNIST, epoch 3 | nudged batches certified | rows certified (epochs 1 / 2 / 3) | sweeps a lesson | accuracy | wall, 3 epochs |
|---|---|---|---|---|---|
| none, master | 0 | - | 32.4 | 96.71% | 14.2 / 12.2 / 12.0 s |
| none, NudgedRate | 0 | 0 / 0 / 0% | 32.4 | 96.71% | 10.7 / 15.1 / 11.0 s |
| spectral, master | 0 | - | 31.8 | 96.98% | 10.3 / 9.4 / 7.5 s |
| spectral, NudgedRate | 0 | 22.6 / 37.1 / 43.8% | 31.8 | 97.07% | 8.2 / 9.0 / 9.8 s |

Reading: the bound is sound and costs nothing measurable (a row that cannot certify leaves as soon as kappa_beta
reaches 1), but it certifies no batch of 32. Under `none` the sup gain grows past 1 (4.5 after an epoch) and nothing
is certified at all. Under the spectral certificate 44% of the rows certify by epoch 3; at the free state alone,
the ball ignored, 77% would - so at kappa 0.9 an uncertain row's `beta lambda_max(S) rho' / T` exceeds `1 - kappa`
outright, and a batch needs all 32 rows. Training is unchanged (no batch certified, so no budget changed; the 0.09
point accuracy difference is the extra warm power iteration `Kappa` advances, which `Restrain` starts from).

## The nudged pair certified in a block norm (wt-cert, 2026-11-17)

`NudgedRate` now takes, row by row, the better of two certificates: the one above (`plainRate`, the certificate's own
norm) and one in the block norm `|x|_t = max(|x_h|, |x_y| / t)` (`blockRate`; h the hidden units, y the readouts, each
part in Bound's norm). A sweep's Jacobian `(1 - dt) + dt (W D + beta J_F)` has its blocks bounded by `M = [a b; c d]`:
`a = sigma_hh s_h`, `b = sigma_hy s_y`, `c = sigma_hy s_h`, `d = sigma_yy s_y + beta f`, each sigma kappa over the
largest slope (a block's norm is at most the whole's) or 0 where no connection joins those units, `s_y` the readouts'
largest slope over the ball and `f` the force's Jacobian over it (`neuron.ForceOver`, between units at +beta, whole at
-beta). In `|.|_t` the sweep contracts by `1 - dt (1 - kappa_t)`, `kappa_t = max(a + b t, c / t + d)`, least at M's
Perron vector `(1, t)`. With no lateral blocks (MNIST's 784-128-10) the condition is `beta f < 1 - kappa^2 s_y / s_h`
instead of `beta f < 1 - kappa`: saturated readouts leave the hidden-readout loop little gain. The ball argument is
the plain one's in `|.|_t` (start residual at most `max(rest, (rest + beta |F|) / t)`, the y-ball's radius `r t`); t is
taken from M at the previous radius and r grown by a quarter until the ball holds. A batch row is certified when either
norm certifies it; the rate says nothing of which norm (the budget only reads q). Tests (`dyn/system.olang`): the old
two pass unchanged, their contraction now measured per row in the norm its rate holds in; a new one on 36 layered
systems (8 -> 32 <-> 10 classes, no lateral block, kappa 0.9, readout bias 1/2/4, beta 0.1/0.5/1, both norms): 13 of
72 rows certified in the plain norm, 34 in the block norm, 21 by it alone, 38 by neither, and every certified row's
step between consecutive states shrinks by at most its rate in its norm, both phases (largest measured ratio 0.88 q;
the test fails if the readouts' slope over the ball is under-counted by 0.3x).

Measured on MNIST (fetched into data/mnist here), `3 60000 128 32 1 0.001 0.5 1 1 0.1 200 1 0 plain spectral`, master
against the branch alternated, 4 runs each, load 15-18 on 4 cores (logs/ in the worktree, not committed):

| MNIST, spectral | rows certified (epochs 1 / 2 / 3) | nudged batches certified | us a lesson, epoch 3 | accuracy |
|---|---|---|---|---|
| master (plain only) | 22.6 / 37.1 / 43.8% | 0 | 1926 / 1934 / 2175 / 2037 | 97.07% |
| block norm too | 35.2 / 53.1 / 60.2% | 0 | 2310 / 2218 / 2360 / 2365 | 97.07% |

Reading: the block norm certifies 16 more points of the rows by epoch 3 (60% against 44%), still no batch of 32 (a
batch needs every row), so no budget changed and training is identical. The bound costs about 10-15% a lesson here
(two row loops of up to 100 growths each, measured under heavy load). The rows left are the uncertain ones: an
unsaturated readout (`s_y` near 1) with a force of size `beta |onehot - p|` near 1 pushes the ball's radius past
where the softmax bounds hold. What is left to try: a budget per row (rows certified given their rate, the rest the
clamped budget - needs a per-row stop in dyn/settle), or a tighter `c` (the hidden slopes weighted by the readouts'
weights, a per-row power iteration).

### Rows that stop leave the products (wt-compact, 2026-11-18)

Built on wt-certrow's per-row stop (below), which saved ~1%. First measured, with an instrumented build, where each row
of an Euclid-measured (spectral) settle first meets the tolerance against the sweep the whole batch stops at (MNIST
`1 60000 128 B 1 0.001 0.5 1 1 0.1 200 1 0 plain spectral`, row-sweeps a stop of each row at its own residual would
leave out): batch 32: 31% of the free and 78% of the nudged (68% overall; the certified rows alone 29% of the nudged);
batch 128: 37 / 86 / 79%; batch 512: 38 / 90 / 85% (certified alone 13%). The rows do not converge together: the batch
runs for its slowest row, most rows are in after a tenth of the sweeps, and the uncertified rows are most of it. The
"rows converge within a few sweeps of each other" read below was wrong - a unit of 8 stopped only when all 8 were
certified and in, and two thirds of the rows were not certified.

So: in a Jacobi, plain, Euclid-measured settle of built-in rate models (`Engine.mayStop`, `Engine.RowStops`, true by
default), free and nudged, **every row** stops at the first sweep its own residual is within the tolerance - the test
the whole batch passes is each row's residual within it, so a row that passes is done whatever the others do (the
certificate, where there is one, bounds its distance as before; an uncertified row is held to what the batch's last row
was always held to). The row keeps the state its sweep wrote, in both buffers, and leaves the work: `Engine.plan`
makes the units the runs of consecutive live rows (at most `Chunk`), re-planned after a sweep in which rows stopped, so
the products run over the live rows only - no data moved, the rows stay where they are. `Engine.Skipped` counts the
row-sweeps left out. wt-certrow's `Certified`/`Stoppable` and its units of 8 are gone. Test: `System.Settle` "a per-row
stop" (36 layered systems: free and nudged settles with and without the stop within 10 tol / (1 - kappa) of each other,
rows stopping at different sweeps, row-sweeps left out).

| MNIST spectral, 1 thread (2 runs, alternated) | row-sweeps left out | us a lesson, master | us a lesson, stop | accuracy master / stop |
|---|---|---|---|---|
| batch 32, epoch 3 (seed 1) | 71% | 2446 / 2430 | 1707 / 1604 | 97.07 / 96.88% |
| batch 32, epoch 3, seeds 2 / 3 | | 2225 / 2453 | 1588 / 1749 | 96.90 / 96.84%, 96.66 / 96.62% |
| batch 128, epoch 1 | 80% | 13787 / 14000 | 5608 / 6199 | 93.70 / 93.85% |
| batch 512, epoch 1 | 85% | 89399 / 95113 | 23531 / 23375 | 92.24 / 92.23% |

1.45x a lesson at batch 32, 2.3x at 128, 3.9x at 512 - a larger batch now costs about what its rows need, not what its
slowest row needs (at 512, 46 us a sample against 175). Accuracy moves by -0.19 to +0.15 points; at batch 32 three seeds
are 0.04-0.19 lower (a stopped row is up to tol / (1 - q) from its fixed point where the batch's later sweeps used to
take it closer). The test answers' free settles stop by row too. Not on the GPU (dyn/mirror sweeps whole batches).

### A per-row stop in the CPU settle (wt-certrow, 2026-11-18; superseded by the entry above)

`System.NudgedRate` now leaves which rows it certified in `Engine.Certified`; the next `System.Nudged` (CPU only; the
GPU branch clears it) makes those rows `Engine.Stoppable`. In a Jacobi, plain (no Anderson), Euclid-measured settle of
built-in rate models, a stoppable row stops at the first sweep its own residual is within the tolerance (`RowStop`
records the sweep): its state is kept in both buffers and a unit of `StopChunk` = 8 rows (instead of 128, only while some
row may stop) whose rows have all stopped is not run again (`Engine.Skipped` counts those row-sweeps). Uncertified rows
keep sweeping to the clamped budget as before. Test: `System.NudgedRate` "a per-row stop" (36 layered systems: rows of
one batch stop at different sweeps, every row within 10 tol / (1 - kappa) of the whole-batch settle, skipped
row-sweeps counted). Measured on MNIST `3 60000 128 32 1 0.001 0.5 1 1 0.1 200 1 0 plain spectral`: 0.0 / 0.4 / 1.4% of
the nudged row-sweeps skipped (epochs 1 / 2 / 3), us a lesson unchanged (2375 at epoch 3 against 2365), accuracy 96.94%
against 97.07% (the stopped rows sit one step apart). Reading: a batch's rows converge within a few sweeps of each other
and a unit stops only when all 8 of its rows have, so the stop saves almost nothing at batch 32; it would pay with
compaction of the live rows into the products (the real lever), not by skipping units.

## The GPU route, GP1: bindings and a GEMM through cuBLAS (wt-gp1, 2026-11-17)

docs/redesign.md 1.10 and 4.4's GP1. `kernels/gpu/cuda.c` (the C shim: device addresses and handles as U64, host arrays
with a byte offset, a status record the Device owns), `kernels/gpu/device.olang` (`Device`, `Buf<S>`, `DMat<S>`,
`GpuError`, uploads and downloads), `kernels/gpu/blas.olang` (`Gemm` by `cublasGemmEx`), `make gputest` and `make
gpugemm` (G-M1, `bench/gpugemm.olang`).

- **The CUDA headers import as they are**: `cuda_runtime_api.h`, `cublas_v2.h`, `cublasLt.h` and `nvrtc.h` through the
  `.c` module, Ubuntu's toolkit 12.0, no declaration refused. OP1 is not needed; OP2 stays worked around by the U64
  addresses, as planned.
- **Gated**: the modules assert `-define Gpu=true` at their top level, so an import of them outside `if Gpu { ... }`
  fails every build saying so, and `make test` translates no CUDA header. `bench/gpugemm` imports them under `if Gpu`
  and times the CPU alone when built without the define.
- **Tests** (`make gputest`, 8): `Device.Open` names the device (NVIDIA GeForce RTX 4060, sm_89, 24 SMs, 8187 MiB);
  upload then download is the identity in F32, BF16, I8 and F64, whole and in ranges at offsets, and an upload is
  ordered before the next compute-stream download; `Gemm` in F32, BF16 (summed in F32) and F64, every transpose,
  alpha and beta, offset views with padded rows, odd sizes (1x1x1 to 100x300x1000, the MLP's and settle's shapes)
  against `kernels/gemm.Gemm` within k eps(A) |alpha| max|a| max|b| + 2 eps(A) |beta| max|c| plus one rounding to S,
  nothing outside C's view written, run twice identical bit for bit; a 4 TiB `Buf` is `GpuError.Cuda` naming
  cudaMalloc and the size with CUDA's "out of memory"; a bad leading dimension is the next Sync's
  `GpuError.Cublas(7, "cublasGemmEx: an unsupported value or parameter was passed to the function")`, and the product
  after it runs. Without a device each test prints that it is skipped.
- **cuBLAS rounds split-K partial sums to BF16 by default**: a BF16 product at depth 257 was 3% off the exact sum (0.0613
  against 0.0597; the CPU 0.0596). The handle is made with `CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` (and the
  default math, so never TF32). Its price is in the table: BF16 128 x 784 x 128 takes 70 us against F32's 15 (no
  split-K left for a 128 x 128 result); from batch 512 BF16 is 1.5-3.2x F32 again. *Corrected by GP3*: the 70 us was
  the bench timing one product and its Sync (its calibration counted cuBLAS's first, kernel-choosing call); warmed,
  it is 8.6 us (GP3's section).
- **This machine** (WSL2): the `nvidia-cuda-dev` package pulled in a native driver's `libcuda.so.1`
  (libnvidia-compute-580) that shadows WSL's in the loader's cache, and CUDA then sees no device. `make gputest` and
  `make gpugemm` put `/usr/lib/wsl/lib` first on `LD_LIBRARY_PATH` where it exists (harmless once the system is fixed);
  run without it, the tests skip.

**G-M1, first table** (`make gpugemm OLANG=olang ARGS="7 1 4"`, 2026-11-17, commit of this section; Ryzen 5 7600 under
WSL2, 6 cores, load average 0.7-1.0; RTX 4060 8 GB, driver 610, CUDA 12.0's cuBLAS). GFLOPS (us a product), the median of
7 interleaved rounds of about 20 ms each. *Resident*: operands on the device, products enqueued back to back, one Sync.
*With transfers*: A and B uploaded from pageable host memory, the product, C downloaded, a Sync - every product; a plan's
per-batch transfers (inputs up, a few numbers down) lie between the two. Each GPU round follows an untimed one: the
GPU drops to 210 MHz (P8) within milliseconds of idling, and a cold BF16 round of 2048^3 measured 3.3 ms against 0.59 ms
warm. Two runs agreed within 10% (BF16's 10 x 128 x 128 within 25%).

F32:

| shape | m x k x n | CPU 1 thread | CPU 4 threads | GPU resident | GPU with transfers |
|---|---|---|---|---|---|
| MLP fwd 1 / settle x->h | 128 x 784 x 128 | 104 (246 us) | 53 (482 us) | 1763 (14.6 us) | 135 (190 us) |
| MLP fwd 2 / settle h->y | 128 x 128 x 10 | 46 (7.2 us) | 46 (7.2 us) | 51 (6.5 us) | 3.1 (105 us) |
| MLP bwd dh / settle y->h | 128 x 10 x 128 | 51 (6.4 us) | 51 (6.4 us) | 52 (6.3 us) | 3.3 (100 us) |
| MLP dW2 / EP outer y h | 10 x 128 x 128 | 49 (6.7 us) | 49 (6.7 us) | 42 (7.8 us) | 3.1 (106 us) |
| MLP dW1 / EP outer h x | 128 x 128 x 784 | 114 (226 us) | 131 (196 us) | 2599 (9.9 us) | 109 (235 us) |
| fwd 1, batch 32 | 32 x 784 x 128 | 78 (82 us) | 17 (372 us) | 493 (13.0 us) | 40 (160 us) |
| fwd 1, batch 512 | 512 x 784 x 128 | 114 (906 us) | 144 (714 us) | 4102 (25.1 us) | 276 (373 us) |
| fwd 1, batch 4096 | 4096 x 784 x 128 | 113 (7278 us) | 346 (2378 us) | 7864 (104 us) | 511 (1609 us) |
| hidden 1024, batch 128 | 128 x 784 x 1024 | 116 (1778 us) | 181 (1133 us) | 6071 (33.9 us) | 405 (508 us) |
| square 512 | 512^3 | 134 (2002 us) | 348 (771 us) | 6829 (39.3 us) | 564 (476 us) |
| square 1024 | 1024^3 | 132 (16297 us) | 451 (4760 us) | 8462 (254 us) | 1476 (1455 us) |
| square 2048 | 2048^3 | 120 (142685 us) | 435 (39473 us) | 9379 (1832 us) | 2769 (6203 us) |

BF16, every operand stored in BF16, summed in F32:

| shape | m x k x n | CPU 1 thread | CPU 4 threads | GPU resident | GPU with transfers |
|---|---|---|---|---|---|
| MLP fwd 1 / settle x->h | 128 x 784 x 128 | 108 (239 us) | 53 (484 us) | 350 (73.4 us) | 175 (147 us) |
| MLP fwd 2 / settle h->y | 128 x 128 x 10 | 53 (6.2 us) | 53 (6.2 us) | 49 (6.7 us) | 3.0 (108 us) |
| MLP bwd dh / settle y->h | 128 x 10 x 128 | 55 (6.0 us) | 55 (6.0 us) | 46 (7.2 us) | 2.6 (125 us) |
| MLP dW2 / EP outer y h | 10 x 128 x 128 | 80 (4.1 us) | 79 (4.1 us) | 33 (10.0 us) | 2.5 (132 us) |
| MLP dW1 / EP outer h x | 128 x 128 x 784 | 119 (215 us) | 159 (162 us) | 1640 (15.7 us) | 103 (249 us) |
| fwd 1, batch 32 | 32 x 784 x 128 | 84 (77 us) | 19 (345 us) | 466 (13.8 us) | 36 (178 us) |
| fwd 1, batch 512 | 512 x 784 x 128 | 110 (938 us) | 145 (707 us) | 6303 (16.3 us) | 404 (254 us) |
| fwd 1, batch 4096 | 4096 x 784 x 128 | 111 (7431 us) | 345 (2380 us) | 16548 (49.7 us) | 871 (944 us) |
| hidden 1024, batch 128 | 128 x 784 x 1024 | 117 (1765 us) | 189 (1090 us) | 13512 (15.2 us) | 576 (357 us) |
| square 512 | 512^3 | 127 (2109 us) | 332 (809 us) | 14249 (18.8 us) | 860 (312 us) |
| square 1024 | 1024^3 | 129 (16683 us) | 426 (5036 us) | 25374 (84.6 us) | 2627 (817 us) |
| square 2048 | 2048^3 | 122 (141229 us) | 447 (38438 us) | 29994 (573 us) | 6053 (2838 us) |

What it says:
- **The settle's 10-wide products stay on the CPU.** 128 x 128 x 10, 128 x 10 x 128 and 10 x 128 x 128 take 4-7 us on
  one core and 6-10 us on the GPU even resident: a cuBLAS launch on WSL costs about 6 us, whatever the work. Anything
  under about 1 MFLOP is launch-bound on the GPU.
- **The 784-wide products go to the GPU once resident**: 128 x 784 x 128 is 15 us against 246 (17x), the EP outer
  product 128 x 128 x 784 10 us against 196-226 (20-23x), and from there the gap grows with the size (2048^3: 78x one
  core, 22x four). The settle on the GPU (GP4) is therefore worth building only with its state resident, as 1.10 plans: a
  sweep is about two big products and launches of about 6 us each.
- **Transfers dominate at these sizes**: a round trip through pageable memory is about 100 us before any work (the 10-wide
  rows), so with nothing resident the GPU wins only from about 128 x 784 x 128 (190 us against the CPU's 246), and
  against four threads by 1.5x at batch 4096 and 6x at 2048^3. Inputs must come up once a batch from pinned staging, as 1.10 says; that is GP2-GP4's.
- **kernels/gemm's four threads were slower than one at 32 and 128 rows - fixed (wt-gemmthreads, 2026-11-17)**: the
  split cost about 300-500 us a product (a spawn and join per block of the product, on this machine's threads), and the
  old rule split anything over 2 M multiply-adds (128 x 784 x 128 is 12.8 M). A product now splits into at most one
  task per 16 M multiply-adds (`madds` in kernels/gemm), none below two. Four threads, bench/storage, best of 3 runs:
  64 x 784 x 128 580 -> 149 us (one thread 146), 128 x 784 x 128 495 -> 252 us (one thread 246); 1024 x 128 x 512
  840 -> 795-1000 and 512^3 and 1024^3 unchanged within the machine's noise (1.5-2.0 ms, 11.5-14 ms), one thread
  unchanged. GemmBatch keeps its own 2 M rule.

## The GPU route, GP2: our own kernels by NVRTC, the fused rate step (wt-gp2, 2026-11-17)

docs/redesign.md 1.10 and 4.4's GP2. `kernels/gpu/kern.olang` (`Kernel.Compile`, `Launch`, `Arg`, `Zero`, `Copy`,
`Timer`; `Axpy`; `Rho` and `Rate<T>`, the settle's fused rate step), `kernels/gpu/src/elem.cu` (FastExp, FastTanh, the
activations, Axpy) and `src/rate.cu` (the fused step, and the same step in three launches for G-M3), the shim's NVRTC,
driver-API, memset/copy and timing-event functions in `cuda.c`, `dyn/gpu_test.olang`, `make gpurate` (G-M3,
`bench/gpurate.olang`).

- **NVRTC at plan time, the PTX cached by hash.** The `.cu` files are embedded while compiling (`embed`) and
  specialized as text: kern.olang writes kernels/elem's FastExp constants and `elem.ExpTaylor` (now public) as their
  bits, then the state's type, the residual word's type and the activation as `#define`s. The PTX is kept at
  `$OLANG_CACHE/oann-ptx/SHA256(NVRTC version, options, text).ptx` (written beside, then renamed); the next compile of
  the same text reads it (`Kernel.Cached`). Measured: NVRTC and the module load 80-106 ms a text (the first compile on a
  cold machine took 1.2 s once), from the cache 0.5-0.8 ms.
- **The driver API through `cudaGetDriverEntryPoint`** (cuModuleLoadData, cuModuleGetFunction, cuModuleUnload,
  cuLaunchKernel, cuGetErrorString), its table owned by each `Kernel`, so the shim keeps no state and nothing links
  libcuda; `cuda.h` is not imported (the five signatures are written in the shim by shape). The PTX is for
  compute_MAJORMINOR of the device and is made sm_89 code by the driver's JIT when loaded.
- **`--fmad=false`**: olang does not contract `a * b + c` (checked: compiled with an `fma` in the step, the test fails
  on Linear's u2 at the first block), so our kernels do not either, and FastTanh on the GPU is kernels/elem's **bit for
  bit** at 2,000,001 points over [-12, 12] plus NaN and the infinities.
- **Launch** packs each argument in a 64-bit slot (an address, or a number's bits; the driver reads a parameter's width
  from the slot's start): tested on Axpy - addresses at offsets, F32 and F64 alphas, a 64-bit count, sizes 1 to
  1,048,579 - bit for bit with kernels/elem's `Axpy`.
- **Errors**: a text that does not compile is `GpuError.Nvrtc` with NVRTC's status and log
  (`NVRTC_ERROR_COMPILATION / oann.cu(1): error: identifier "undeclared_thing" is undefined ...`); a kernel the module
  lacks is `GpuError.Launch` (`cuModuleGetFunction(oann_nothing): named symbol not found`); a launch the driver refuses
  (a block of 2048 threads) is recorded in the status record (a new kind, the driver's text kept) and given back by the
  next `Sync` as `GpuError.Launch("cuLaunchKernel: invalid argument")`; the launch after it runs.
- **The rate step** (`Rate<T>.Make(dev, rho)`, `Step`): r = total - u, the residual's |r| bits into a device word by
  one `atomicMax` a block (warp shuffles first), u2 = u + dt r with dt per phase (an array, the rows phase-major),
  s2 = rho(u2), retired units (a byte mask per unit) put at 0 - dyn/settle's rest-then-retire, fused. Against
  `neuron.RateStep` with the settle's retired-unit handling (`dyn/gpu_test.olang`, F32 and F64, every Rho, the settle's
  shapes, odd sizes, a 1024 x 512 x 2 block, offsets, retired units, special values: +-0, +-1e30, +-1e-30, +-Inf):
  u2 and s2 bit for bit for Linear, Relu, HardSigmoid - and for Tanh too (asserted within FastTanh's 2.0e-7, measured
  bit for bit at every element); LifRate bit for bit in F32, in F64 at all but 449 of 1,048,576 in the largest case (the GPU's `log`
  against the C library's; within the test's 2 ulp); the residual word the CPU's sup norm bit for bit, and a NaN wherever the
  CPU's is one (a NaN in u or in the totals, at block edges, beside a residual of 1e30); two launches into one word give
  the one launch's maximum; run twice identical.
- **Not done here** (GP4's): the converged word that stops later sweeps' steps, the readout's force kernel, device
  mirrors of a plan's state.

**G-M3** (`make gpurate OLANG=olang ARGS=7`, 2026-11-17, commit of this section; RTX 4060 under WSL2, Ryzen 5 7600). The
median of 7 rounds, each after an untimed one. *Back to back*: CUDA events around many reps enqueued as fast as the host
goes. *Queued ahead*: the same behind 20 ms of other GPU work, so every launch is queued before the first runs and the
events see the GPU alone (checked: the enqueueing ended within the blocker every time). *Host*: the host's time to
enqueue one. Tanh, F32.

The step alone, us a step - queued ahead / back to back / host:

| population | elements | fused, 1 launch | unfused, 3 launches | unfused / fused (queued) |
|---|---|---|---|---|
| 10 x 128 rows | 1280 | 6.53 / 5.30 / 4.84 | 17.78 / 16.40 / 16.42 | 2.72 |
| 128 x 128 rows | 16384 | 6.43 / 5.78 / 3.76 | 17.44 / 15.36 / 11.38 | 2.71 |
| 10 x 256 rows | 2560 | 5.91 / 5.15 / 4.08 | 16.74 / 15.69 / 15.91 | 2.83 |
| 128 x 256 rows | 32768 | 6.13 / 5.62 / 3.91 | 17.33 / 16.26 / 11.29 | 2.83 |
| 128 x 512 rows | 65536 | 6.35 / 5.89 / 3.83 | 22.00 / 16.94 / 11.25 | 3.46 |
| 1024 x 512 rows | 524288 | 14.49 / 13.26 / 4.01 | 28.01 / 26.63 / 11.70 | 1.93 |
| 1024 x 4096 rows | 4194304 | 269.93 / 273.24 / 3.99 | 527.97 / 528.95 / 12.84 | 1.96 |
| 4096 x 8192 rows | 33554432 | 2200.88 / 2183.57 / 3.97 | 4338.64 / 4333.62 / 13.36 | 1.97 |

A sweep of the 784-128-10 settle at batch 128 as GP4 would enqueue it (the residual word zeroed; per population its
constant copied into its totals, one copy a phase, its carry by cuBLAS with beta 1, its rate step), GPU us a sweep:

| | launches | queued ahead | back to back | host to enqueue | with the word read back and waited for (wall) | the floor: as many launches of a one-element Axpy, back to back |
|---|---|---|---|---|---|---|
| free phase (128 rows), fused | 7 | 43.3 | 37.6 | 37.9 | 87.5 | 36.6 |
| free phase, unfused | 11 | 70.3 | 54.6 | 57.3 | 108.2 | 52.4 |
| nudged pair (256 rows), fused | 9 | 61.3 | 47.2 | 49.5 | 105.1 | 46.2 |
| nudged pair, unfused | 13 | 77.6 | 73.0 | 73.6 | 151.9 | 68.6 |

Per launch group (free phase, fused, back to back): the word's zeroing 5.0 us, a constant's copy 5.0-5.3, the carries
6.4 and 6.6 (cuBLAS), the rate steps 5.3 and 5.1. The rate steps are 27% of the sweep fused (2 of 7 launches), 52%
unfused (6 of 11).

What it says:
- **A launch costs 5-7 us on this machine whatever it does**, on the host (4-5 us to enqueue) and on the GPU alike: queued
  ahead, with no host in the way, a launch of a one-element kernel still takes 6 us of GPU time (WSL2's submission
  path, presumably; not measured on a native Linux driver). At the settle's sizes the work itself does not show: a sweep is
  94-98% its floor of empty launches, so **the number of launches is the sweep's time**.
- **Fusing the step pays exactly its launches**: 2.7-3.5x for the step alone at the settle's sizes (one launch for
  three), a sweep 1.45x faster fused (37.6 us against 54.6, 47.2 against 73.0). From half a million elements, where
  the memory decides, fused is 1.93-1.97x (five arrays read or written against eight, and a pass more).
- **Waiting for the residual each sweep doubles a sweep** (87.5 us wall against 37.6): the residual check every `Check`
  sweeps (1.10) is needed, as planned.
- **CUDA graphs and further fusion are worth a package (GP7)**: a fused sweep is 7 launches (9 with two phases), at
  least 4 of them avoidable by fusion alone - the constants' copies folded into the step (the step reads the constant
  and the carry, the product written with beta 0) or into cuBLASLt's C != D (GP3), and the word's zeroing into the
  previous sweep's check - leaving 2 products and 2 steps (at the 5-6.6 us a launch measured above, 20-26 us a sweep,
  derived, not measured). Whether a graph's replay lowers the
  GPU-side 6 us a node on WSL2 is to be measured there; on this machine 37.6 us a sweep is the GPU's floor at batch
  128 until then, against the CPU's sweep in G-M2 (GP5).

## The GPU route, GP3: products through cuBLASLt (wt-gp3, 2026-11-17)

docs/redesign.md 1.10 and 4.4's GP3. `kernels/gpu/blas.olang` (`Product<SC, SA, SB>` with `Plan`, `PlanBatch`, `Run`,
`RunBatch`, `Tune`; `Epilogue<S>`, `Sums`, `Choice`; `ProductI8` with `Plan`, `Run`, `RunI32`, `Tune`), `DBatch<S>`
(`Heads`, `Stacked`, `At`) and the `Lt`/`Work` accessors in `device.olang`, the shim's cuBLASLt block in `cuda.c`
(plans, runs, tuning, widening by `cublasLtMatrixTransform`, `cublasSdgmm`, stream capture), `make gpult` (G-M1 on
cuBLASLt, INT8 and attention, `bench/gpult.olang`).

- **Plans.** A `Product` is cuBLASLt's matmul descriptor, three layouts and the algorithm its heuristic offered first,
  all made at `Plan` from views at the addresses they will run on: the plan keeps the alignment the addresses share
  (up to 256 bytes) and tells the heuristic, and a run on an address less aligned is refused into the status record
  (the next `Sync`'s `GpuError.Cublas(15, "cublasLtMatmul: an address less aligned than its plan's ...")`), never run.
  `Tune` times every algorithm offered on the plan's views (events, 20 products each) and keeps the fastest; `Tried`
  keeps what it measured. Run is two C calls; `assert not Product<F32, F32, F32>.Run.Allocates` (and RunBatch's,
  ProductI8.Run's) holds while compiling, and on the device two plans' runs captured into a CUDA graph (thread-local
  capture mode: an allocation would fail it) replay to the same bits. The workspace is the Device's 32 MiB, shared with
  the cuBLAS handle.
- **Sums, a plan-time choice.** `Exact` (the default): `CUBLASLT_REDUCTION_SCHEME_NONE`, so no split-K, every element
  one F32 sum rounded once, and a product run twice is the same bit for bit (tested). `Fast`: any reduction allowed,
  for BF16 the parts possibly added in BF16. `FastBF16`/`FastTF32` for F32 storage: `CUBLAS_COMPUTE_32F_FAST_16BF`/
  `_TF32`. Each has a stated bound in the tests (Exact eps(F32) k max|a||b| + C's rounding; Fast adds SplitK roundings
  of C's type; FastBF16 adds eps(BF16), FastTF32 2^-9 per unit of k max|a||b|), checked with the fast plans tuned.
- **Epilogues**: `Bias` and `BiasAct` with Relu and GeluTanh are cuBLASLt's `BIAS`, `RELU(_BIAS)`, `GELU(_BIAS)`;
  cuBLASLt's GELU is the tanh form (5e-6 from it on [-4, 4], 5e-4 from the exact one), so `Gelu`, `Silu`, `Sigmoid`,
  `Tanh` and `Softplus` are refused at `Plan` naming the act (`GpuError.Cuda(0, "no GPU form of the epilogue's ...")`).
  The bias is stored in C's type (cuBLASLt takes no other), and for a BF16 C the sum is rounded to BF16 before the bias
  is added (found: 5 units off where the bias cancels the sum; the test's bound says so).
- **Per-operand storage**: cuBLASLt multiplies BF16 x BF16 and F16 x F16 into their own type or F32, and no mixed
  pair (F32 x BF16, BF16 x F16: no algorithm, FP8 aside). A mixed plan widens the narrower operand into a buffer of its
  own by `cublasLtMatrixTransform` (BF16 and F16 to F32, exact) each run, then multiplies in F32 - tested against
  `kernels/gemm.Gemm<F32, F32, BF16, F32>` and the other pairs. Its cost is a second launch (below): weights that do not
  change between runs are better kept widened, or the activations narrowed to BF16 (`Product<F32, BF16, BF16>`).
- **GemmBatch** (`PlanBatch`/`RunBatch` over `DBatch`): one call over every matrix where each operand's two strides fold
  into one (attention's stacked scores), else one call per member or group with the other as cuBLASLt's batch
  (attention's `Heads`, whose two strides do not fold). cuBLASLt has no triangular product - its `FILL_MODE` is
  accepted and then no algorithm is offered for LOWER or UPPER - so a causal product is the full one: the scores'
  upper triangle is computed and the softmax that reads them masks it, and P V multiplies P's zeros. That mask is
  GP2/GP6's softmax kernel's, not a kernel of its own. Tested on attention's shapes (16 x 64, width 128, 4 heads; 2 x
  33, 96, 3; 3 x 17, 40, 1; 1 x 128, 256, 8) in F32 and BF16 against `GemmBatch` (Full for the scores, `Left(0)` for P
  V with P causal), run twice identical.
- **INT8** (`ProductI8`): c = a b^T by cuBLASLt's integer tensor cores (`CUBLAS_COMPUTE_32I`, I8 x I8 -> I32), exact
  (equal to the integer sums in the test). out = scale[j] c needs I32 -> F32, and cuBLAS has no such conversion:
  `cublasLtMatrixTransform` refuses every integer type (status 15), and the per-row alpha vector exists only for I8
  results. So out is a second product summed in F32 (`COMPUTE_32F` over I8, which runs without tensor cores) and a
  `cublasSdgmm`. Tested against `kernels/gemm.GemmI8` (U8 activations, zero point 128) within a quantization step:
  |out_gpu - out_cpu| <= s sw[j] sum_p |q[j, p]|, at 1 to 2048 rows, depths 12 to 2048 (multiples of 4). A one-pass
  dequantization kernel (now possible with GP2's kern.olang) replaces the second product.
- **GP1's BF16 number was a measurement artifact**, not cuBLAS: bench/gpugemm calibrated its repetitions on the shape's
  first product, which includes cuBLAS choosing its kernel (over 20 ms), so a round was one product with its `Sync`'s
  round trip. Warmed first (fixed), cuBLAS's own BF16 128 x 784 x 128 with `DISALLOW_REDUCED_PRECISION_REDUCTION` takes
  8.6 us against F32's 16.6 (`make gpugemm OLANG=olang ARGS="5 1 4"`, rerun): GP1's `Gemm` stays as it is.

**G-M1 on cuBLASLt** (`make gpult OLANG=olang ARGS=5`, 2026-11-17, commit of this section; the machine as GP1's table,
load average 1.1-3.0). us a product, every operand resident, the median of 5 interleaved rounds of about 20 ms each
after an untimed one. CPU: kernels/gemm, one thread, F32. cuBLAS: GP1's `Gemm`. The rest are `Product`s: F32 and BF16
Exact by the heuristic's algorithm and tuned; BF16 fast (`Sums.Fast`, tuned); BF16>F32 (BF16 operands, C in F32, Exact,
tuned); F32 via BF16 and TF32 (`FastBF16`, `FastTF32`, tuned); F32 x BF16 W (BF16 weights widened each run, tuned).

| shape | m x k x n | CPU | cuBLAS F32 | cuBLAS BF16 | F32 | F32 tuned | BF16 | BF16 tuned | BF16 fast | BF16>F32 | F32 via BF16 | F32 via TF32 | F32 x BF16 W |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP fwd 1 / settle x->h | 128 x 784 x 128 | 251 | 14.5 | 8.6 | 14.9 | 14.9 | 8.7 | 8.6 | 8.6 | 8.7 | 9.7 | 12.8 | 19.3 |
| MLP fwd 2 / settle h->y | 128 x 128 x 10 | 7.3 | 6.6 | 5.7 | 6.7 | 6.7 | 5.7 | 5.6 | 5.8 | 5.8 | 6.7 | 6.6 | 13.0 |
| MLP bwd dh / settle y->h | 128 x 10 x 128 | 6.0 | 6.1 | 5.5 | 6.2 | 6.1 | 5.5 | 5.5 | 5.5 | 5.4 | 6.1 | 6.1 | 11.5 |
| MLP dW2 / EP outer y h | 10 x 128 x 128 | 6.8 | 6.8 | 6.2 | 6.7 | 6.8 | 6.2 | 6.2 | 6.3 | 6.4 | 6.9 | 6.7 | 13.2 |
| MLP dW1 / EP outer h x | 128 x 128 x 784 | 228 | 9.6 | 8.4 | 9.6 | 9.8 | 8.4 | 7.5 | 7.5 | 7.8 | 9.7 | 8.8 | 17.3 |
| fwd 1, batch 32 | 32 x 784 x 128 | 83 | 12.9 | 11.4 | 14.8 | 14.7 | 14.8 | 9.0 | 11.3 | 9.1 | 9.1 | 12.4 | 19.1 |
| fwd 1, batch 512 | 512 x 784 x 128 | 959 | 26.7 | 15.3 | 28.3 | 28.1 | 15.5 | 9.2 | 9.1 | 9.4 | 15.6 | 18.2 | 33.3 |
| fwd 1, batch 4096 | 4096 x 784 x 128 | 7636 | 122 | 58.8 | 151 | 116 | 54.5 | 46.6 | 47.0 | 48.7 | 52.9 | 87.7 | 118 |
| hidden 1024, batch 128 | 128 x 784 x 1024 | 1795 | 36.4 | 15.3 | 46.8 | 48.0 | 15.2 | 14.9 | 14.9 | 15.0 | 21.8 | 25.3 | 58.7 |
| square 512 | 512^3 | 2116 | 38.6 | 20.5 | 47.8 | 51.0 | 20.5 | 17.4 | 17.2 | 17.8 | 20.3 | 28.9 | 52.0 |
| square 1024 | 1024^3 | 18452 | 263 | 102 | 289 | 274 | 94.0 | 88.8 | 85.8 | 85.9 | 103 | 175 | 293 |
| square 2048 | 2048^3 | 156210 | 1965 | 670 | 2013 | 2128 | 689 | 625 | 597 | 595 | 693 | 1311 | 1973 |

The error of each way against the exact product (F64, kernels/gemm) of the numbers it was given - operands uniform in
[-1, 1), F32 for the F32 ways, rounded to BF16 for the BF16 ways: max|err| / max|exact|, then max|err| / (k eps(BF16)
max|a| max|b|), eps(BF16) = 2^-7. Rows: 128 x 784 x 128, 512 x 784 x 128, 2048^3 (the bench prints all twelve; the
others are alike):

| way | 128 x 784 x 128 | 512 x 784 x 128 | 2048^3 |
|---|---|---|---|
| F32 (Exact; tuned alike; cuBLAS within 2x) | 3.7e-7 / 2.5e-6 | 4.0e-7 / 2.4e-6 | 1.9e-6 / 9.7e-6 |
| BF16 (Exact, tuned, Fast, cuBLAS's: all alike) | 2.8e-3 / 1.9e-2 | 3.3e-3 / 2.0e-2 | 3.0e-3 / 1.5e-2 |
| BF16>F32 | 1.5e-6 / 1.0e-5 | 1.1e-6 / 6.5e-6 | 3.5e-6 / 1.8e-5 |
| F32 via BF16 | 1.8e-3 / 1.2e-2 | 2.3e-3 / 1.4e-2 | 2.1e-3 / 1.1e-2 |
| F32 via TF32 | 2.6e-4 / 1.7e-3 | 2.9e-4 / 1.7e-3 | 2.6e-4 / 1.3e-3 |
| F32 x BF16 W | 3.6e-7 / 2.4e-6 | 4.0e-7 / 2.4e-6 | 2.0e-6 / 1.0e-5 |

INT8 and attention (same run; us, resident):

| GemmI8 m x k x n | CPU GemmI8 (1 thread) | GPU c only (I32) | GPU c and out (F32) | GPU BF16 tuned |
|---|---|---|---|---|
| 1 x 784 x 128 | 3.1 | 5.8 | 17.8 | 5.7 |
| 32 x 784 x 128 | 33.5 | 6.5 | 22.6 | 9.5 |
| 128 x 784 x 128 | 75.4 | 7.3 | 23.9 | 9.5 |
| 512 x 784 x 128 | 264 | 10.6 | 40.6 | 8.9 |
| 1 x 768 x 3072 | 65.4 | 7.9 | 20.9 | 9.4 |
| 64 x 768 x 3072 | 1121 | 8.7 | 60.9 | 15.2 |
| 2048^3 | 37986 | 192 | 2466 | 586 |

| attention, B x T, width, heads | CPU F32 causal (1 thread, triangles) | GPU F32 (full) | GPU BF16 (full) |
|---|---|---|---|
| 16 x 64, 128, 4 (the character transformer) | 463 | 58.3 | 44.6 |
| 8 x 256, 512, 8 | 7445 | 279 | 131 |
| 4 x 1024, 768, 12 | 89930 | 2125 | 996 |

What it says:
- **BF16 is faster than F32 at every shape, at full accuracy.** Exact BF16 (F32 sums, no split-K, no BF16-rounded
  partial sums) tuned against F32 tuned: 1.7x at 128 x 784 x 128 (8.6 against 14.9 us), 1.6x at batch 32, 3.1x at
  batch 512, 2.5x at batch 4096, 3.2x at hidden 1024, 2.9x at 512^3, 3.1x at 1024^3, 3.4x at 2048^3; the 10-wide
  products are launch-bound (5.5-6.2 against 6.1-6.8 us) and still a little ahead. Its only error beyond F32's is C's
  rounding to BF16 (2-3e-3 relative, half a unit of BF16); with C in F32 (BF16>F32) it is as fast and 1e-6 off.
- **Fast buys nothing here**: tuned, it is the exact plan's time within noise at every shape (2048^3: 597 against 625,
  the same algorithm, no split-K, chosen by both), and at batch 32 Tune's pick (split-K 12) ran slower in the table
  than the exact plan (11.3 against 9.0). cuBLASLt's split-K helps only where the output is small and k long, which
  these shapes are not. Recommendation: **`Sums.Exact` stays the default**; GP6 can confirm training is unaffected.
- **Tuning pays where the heuristic guesses wrong**: BF16 at batch 512, 15.5 -> 9.2 us; F32 at batch 4096, 151 -> 116.
  A plan should be tuned once where its operands exist.
- **F32 through BF16 tensor cores is a worse BF16**: as slow as BF16 storage or slower and as inaccurate (the operands'
  rounding, 2e-3). Through TF32: 1.2-1.9x F32's speed at 2.6e-4. Neither is worth a default.
- **BF16 weights under F32 activations** cost a widening launch each run (+5-6 us, and a pass over the weights): slower
  than F32. Better: keep such weights widened (they are constant in inference) or narrow the activations (BF16>F32).
- **INT8's I32 product is the fastest product here** (2048^3: 192 us, 3x BF16, 10x F32), but its F32 out, without a
  conversion in cuBLAS, costs a second product that runs without tensor cores (2466 us). G-M5 waits for a one-pass
  dequantization kernel on GP2's kern.olang.
- **Attention's two products** on the GPU are 8-90x the CPU's one thread even computing the full squares the CPU's
  triangles skip; BF16 halves F32's time from T = 256.

## The GPU route, GP4: the settle on the GPU (wt-gp4, 2026-11-17)

docs/redesign.md 1.10 and 4.4's GP4. `kernels/device.olang` (the backend, chosen once: `Backend`, `SetDefault`,
`GpuAvailable`, `DeviceError`; the program's device), `dyn/mirror.olang` (a planned system's regions on the device, the
settle, EP's contrast and AdamW's step there), `kernels/gpu/src/settle.cu` (the fused step with the constant, the carries
and the readout's nudge; the stop; the rows' Euclidean residual; closed forms, constants, EP's differences and bias
sums), elem.cu's `oann_adamw` and `kern.AdamW`, the shim's pinned staging (its own block in `cuda.c`), `System.Plan`'s
and `Graph.Plan`'s `on`, `learn.AdamW`, `dyn/gpu_test.olang`'s GP4 tests, `make gpusettle` (`bench/gpusettle.olang`).

- **The backend, chosen once.** A script writes
  `device.SetDefault(device.Backend.Gpu if device.GpuAvailable() else device.Backend.Cpu)` (module `kernels/device`,
  imported with or without the define) and every plan after it takes that backend unless its `on` says otherwise; the
  device is the module's global, opened by the first GPU plan. `Graph.Plan` on the GPU is refused, saying so (GP6). The
  makefile passes `-define Gpu=true` wherever `/usr/include/cuda_runtime_api.h` exists (`GPU=0` leaves it out), so `make
  test` here builds every module with the GPU and still runs only the CPU's tests: **362 passed, 0 failed**.
- **A sweep is 4 launches** for 784-128-10 (GP2's G-M3 sweep was 7): per relaxing population one cuBLAS product a carry
  over all the settle's rows and one fused step that adds the constant (no copy), the carries and the nudge itself; the
  words are zeroed once a settle. The host enqueues `Check` sweeps (4) and reads the previous batch's words after an
  event into pinned memory, so it never waits for a sweep it has not passed; the step of the sweep after a converged one
  records it in the stop word and writes nothing, nor does any after it, so the state kept is relax's.
- **Tests** (`make gputest`, **34 passed**; GP4's 8 in `dyn/gpu_test.olang`): 784-128-10 lessons (free settle, nudged
  pair, contrast) CPU against GPU from the same parameters and state, batch 32 (6 lessons) and 128 (4), `Check` 4, 1
  and 16, F32: the same sweeps every settle, the same certified rate and rows, the fixed points at most 9e-7 apart
  (asserted within tol / 50, tol 1.25e-3: a sweep past the converged one would move them by about tol), EP's estimate at
  most 4.5e-8 apart; F64 the same, 1.1e-15 apart. Dense, Symmetric and LowRank (lateral and folded) connections,
  retired units, a Euclidean budget, F32 and F64: the same sweeps, the states within the tolerance. EP's contrast on the
  same states: 3e-8 from the CPU's (largest 0.07). AdamW on the GPU is nn/optim's **bit for bit** (and `learn.AdamW` on the
  CPU too), F32 and F64, with decay, the rate changed between steps. Refused at `Plan` by name: spiking populations,
  Csr, Tiles, Events links, Anderson, Gauss-Seidel. The default backend: `SetDefault(Gpu)` makes `Plan`'s default the
  GPU, an explicit `on` still decides. Allocation: pinned while compiling - `System<F32>.Settle` and `Nudged` (now
  through the GPU path too), the mirror's settle, contrast, AdamW step, Reset and Warm, `learn.AdamW.Step`,
  `Learner.Contrast` - and the address space unchanged (VmPeak) over 300 GPU lessons with AdamW's steps.
- **dyn/system's flaky growth check** ("a lesson allocates nothing", the resident set over 400 settles) failed on every
  run here, base included (70 pages), while the GP3 agent's builds shared RAM: transparent huge pages are `always`.
  It now measures VmPeak, as olang's growth checks do: 0 KB over the 400 settles, asserted 0.
- **olang**: a function returning a reference to a module's global (`return opened!`, the device) called from
  `System<T>.Plan` gave `error[O10c]: the callee needs one argument's scope to outlive another's` at every builder that
  plans a system it returns (`System<T>.tight`, `.mixed`) - for a callee with no arguments. Small programs of the same
  shape (generic receiver, global under `if Gpu`, unique type with a destructor, try/catch converting the error) compile;
  not reduced to a repro. Worked around: `device.Open()` fills the public global `device.Opened`, read by `Plan`.
  And by design (S8a): in a build without the define a `GpuAvailable()` returning the literal `false` made a script's
  `if device.GpuAvailable()` a branch the build decided, refused as dead code; it returns a global never set instead,
  so the script line compiles in both builds (checked: a program with the line and an `if` on it, built both ways).

**First timings** (`make gpusettle OLANG=olang ARGS="5 10 20 4"`, 2026-11-17, commit of this section; Ryzen 5 7600 under
WSL2, load average 1.3-1.6, RTX 4060). The 784-128-10 system of mnist_learn on MNIST-like batches drawn in the bench
(sparse pixels, random classes - not MNIST, so its settles are short: 4-5 sweeps cold, 8 a lesson); a cold free settle
(Reset, Settle to the lesson tolerance 1.25e-3) and a lesson (EP's free and nudged settles and contrast, AdamW's step),
the median of 5 interleaved rounds of 10 settles and 20 lessons each, after an untimed round; the GPU's include every
transfer. Two runs agreed within 5%.

| batch | backend | us a cold settle | sweeps | us a lesson | sweeps a lesson | settle / GPU | lesson / GPU |
|---|---|---|---|---|---|---|---|
| 32 | CPU 1 thread | 125 | 4.4 | 351 | 8.4 | 0.23 | 0.30 |
| 32 | CPU 4 threads | 449 | 4.4 | 668 | 8.4 | 0.81 | 0.57 |
| 32 | GPU | 552 | 4.4 | 1168 | 8.4 | 1 | 1 |
| 128 | CPU 1 thread | 410 | 4.7 | 1146 | 7.9 | 0.80 | 1.04 |
| 128 | CPU 4 threads | 644 | 4.7 | 1347 | 7.9 | 1.26 | 1.23 |
| 128 | GPU | 510 | 4.7 | 1099 | 7.9 | 1 | 1 |
| 512 | CPU 1 thread | 1498 | 4.0 | 4513 | 7.9 | 2.21 | 3.31 |
| 512 | CPU 4 threads | 1424 | 4.0 | 3969 | 7.9 | 2.11 | 2.91 |
| 512 | GPU | 676 | 4.0 | 1362 | 7.9 | 1 | 1 |

What it says, first (GP5 measures it on MNIST, with `Check` 1, 4, 16 and hidden 1024):
- **The GPU's settle is flat in the batch** (510-680 us from 32 to 512 rows) and the CPU's linear: the CPU wins at 32
  (one thread 4.4x faster), they meet at 128, and at 512 the GPU is 2.1-2.3x faster a settle and 2.9-3.4x a lesson.
- **At these sweep counts the GPU's time is its fixed cost**, not its sweeps: 4-5 sweeps of 4 launches (6 us each, G-M3)
  are about 120 us of a 550 us settle; the rest is the batch's staged upload, the closed form and the constant's
  product, the readbacks (at least two event waits a settle) and the commit. A lesson adds the nudged settle, the host's
  certificate (the coupling and the readout's state down, `Kappa`, `NudgedRate`) and the contrast. MNIST's longer settles
  will weigh the sweeps more; CUDA graphs or fewer readbacks are GP7's if G-M2 says so.
- kernels/gemm's four threads are again slower than one below batch 512 (G-M1's finding).

## The GPU route, GP5: G-M2, where the settle runs (wt-gp5, 2026-11-17)

docs/redesign.md 1.10's G-M2 and 4.4's GP5. `examples/mnist_learn.olang` takes the backend as its 8th argument: a thread
count (the CPU), `gpu` (the GPU, set with `device.SetDefault`, residual words read every 4 sweeps), `gpuN` (every N:
the mirror's `Check`) or `auto` (the rule below); it now steps `learn.AdamW` (nn/optim's step for step: the CPU's
output is unchanged to the digit). `System.Hold` with nothing masked does nothing - on the GPU it had brought every
parameter down and up again each lesson. `bench/gpulearn.olang` (`make gpulearn`) runs the program over the grid and
takes the medians. MNIST lives in `/home/oscar/claude/data/mnist`, linked as `data/mnist` (`.gitignore` takes the link).

**Verdict.**
- **The settle runs on the GPU where batch x hidden is at least 16384, else on the CPU at one thread.** Measured: at
  16384 (128 x 128, 64 x 256, 32 x 512) the GPU is 1.12-1.24x faster an epoch than the best CPU; at 8192 (64 x 128,
  32 x 256) the CPU is 1.5-1.7x faster, at 4096 (32 x 128) 3.0x. Above it the GPU's lead grows with the work: 2.2x at
  64 x 512, 2.8x at 32 x 1024, 3.4x at 512 x 128, 6.1x at 128 x 1024, 8.9x at 512 x 1024. So for mnist_learn's default
  (batch 32, hidden 128) the CPU stays; at hidden 1024 the GPU wins from batch 32. `system.SettleOn(batch, width)` is
  this rule in one line (and `mnist_learn ... auto`); `Plan`'s default stays the CPU. It holds on the rebased tree
  (below), at 16384 by 1.04-1.16x.
- **`Check` 4**, the default, is the fastest in every cell: `Check` 1 (a host wait every sweep) costs 1.2-1.5x, `Check`
  16 (up to 15 wasted sweeps a settle, at 26-55 sweeps a lesson) 1.1-1.25x. Sweeps a lesson are the same for all three.
- **On the CPU one thread** wherever the GPU loses: 4 threads are slower in every cell below 16384 (1.2-1.5x; GP4's and
  G-M1's finding); without a GPU, 4 threads win from 512 x 128 (1.26x) and at hidden 1024 from batch 128 (1.7-2.8x).
- **Accuracy matches.** CPU and GPU start from the same parameters; at batch 128 and 512 the test accuracy is the same
  to the digit for nearly every seed. At batch 32 (1875 lessons an epoch) the runs drift apart by rounding and land on
  different but equally good points: over 15 seeds the medians are 97.12% (CPU) and 97.12% (GPU) at hidden 128, 96.11%
  and 96.04% at hidden 1024; the mean GPU - CPU difference per seed -0.06 and -0.03 points, against seed ranges of
  96.47-97.39% and 94.78-96.74%. The GPU is reproducible: seeds 1-5 gave the same accuracies in both sessions.
  `Check` changes nothing (1, 4, 16: the same accuracy, the same sweeps).
- **What an epoch costs** (60000 samples, the lessons only; median, s): hidden 128 - CPU 1.28 / 1.55 / 1.96 at batch
  32 / 128 / 512 (one thread), GPU 3.87 / 1.25 / 0.45; hidden 1024 - CPU 11.67 / 14.16 / 21.36 (one thread), 12.27 /
  8.49 / 7.57 (four), GPU 4.20 / 1.39 / 0.85. The GPU's free settle of a test batch is about 0.9-1.5 ms at every size
  (the CPU's 0.19 ms at 32 x 128, 24 ms at 512 x 1024): its fixed cost a settle, not its sweeps, sets the crossover -
  GP7's CUDA graphs and fewer readbacks would move it down.

**The table** (`make gpulearn OLANG=olang ARGS="5 3"`, commit 7416042: 3 epochs a run, 5 interleaved rounds a cell -
each backend once a round, the order rotated, seed = the round - the medians over rounds of a run's mean epoch; the
accuracy after epoch 3, median and the 5 seeds' range; a test settle is the free settle of one test batch. Ryzen 5 7600
(6 cores) under WSL2, RTX 4060; load average 1.0-1.5 while running (0.5 idle before), nothing else built or run. Logs
in `logs/gm2*.log`, not in git):

| hidden | batch | backend | s an epoch | sweeps a lesson | test accuracy % (seeds) | us a test settle | / the cell's fastest |
|---|---|---|---|---|---|---|---|
| 128 | 32 | CPU 1 thread | **1.28** | 31.3 | 96.87 (96.47-97.13) | 194 | 1.00 |
| 128 | 32 | CPU 4 threads | 1.89 | 31.3 | 96.87 (96.47-97.13) | 511 | 1.48 |
| 128 | 32 | GPU, Check 1 | 6.00 | 31.3 | 96.71 (96.47-96.91) | 1585 | 4.67 |
| 128 | 32 | GPU, Check 4 | 3.87 | 31.3 | 96.71 (96.47-96.91) | 898 | 3.02 |
| 128 | 32 | GPU, Check 16 | 4.73 | 31.3 | 96.71 (96.47-96.91) | 1165 | 3.69 |
| 128 | 128 | CPU 1 thread | 1.55 | 45.3 | 96.60 (96.57-96.79) | 650 | 1.24 |
| 128 | 128 | CPU 4 threads | 1.73 | 45.3 | 96.60 (96.57-96.79) | 912 | 1.38 |
| 128 | 128 | GPU, Check 1 | 1.89 | 45.3 | 96.60 (96.56-96.79) | 1305 | 1.51 |
| 128 | 128 | GPU, Check 4 | **1.25** | 45.3 | 96.60 (96.56-96.79) | 946 | 1.00 |
| 128 | 128 | GPU, Check 16 | 1.41 | 45.3 | 96.60 (96.56-96.79) | 1174 | 1.13 |
| 128 | 512 | CPU 1 thread | 1.96 | 54.4 | 95.27 (95.15-95.38) | 2395 | 4.35 |
| 128 | 512 | CPU 4 threads | 1.55 | 54.4 | 95.27 (95.15-95.38) | 2285 | 3.44 |
| 128 | 512 | GPU, Check 1 | 0.68 | 54.4 | 95.27 (95.15-95.38) | 1420 | 1.50 |
| 128 | 512 | GPU, Check 4 | **0.45** | 54.4 | 95.27 (95.15-95.38) | 1105 | 1.00 |
| 128 | 512 | GPU, Check 16 | 0.53 | 54.4 | 95.27 (95.15-95.38) | 1314 | 1.17 |
| 1024 | 32 | CPU 1 thread | 11.67 | 26.2 | 96.19 (96.00-96.38) | 2229 | 2.78 |
| 1024 | 32 | CPU 4 threads | 12.27 | 26.2 | 96.19 (96.00-96.38) | 2458 | 2.92 |
| 1024 | 32 | GPU, Check 1 | 5.95 | 26.2 | 95.76 (95.55-96.49) | 1585 | 1.42 |
| 1024 | 32 | GPU, Check 4 | **4.20** | 26.2 | 95.76 (95.55-96.49) | 1043 | 1.00 |
| 1024 | 32 | GPU, Check 16 | 5.14 | 26.2 | 95.76 (95.55-96.49) | 1298 | 1.22 |
| 1024 | 128 | CPU 1 thread | 14.16 | 34.7 | 96.37 (96.32-96.62) | 6692 | 10.21 |
| 1024 | 128 | CPU 4 threads | 8.49 | 34.7 | 96.37 (96.32-96.62) | 5466 | 6.12 |
| 1024 | 128 | GPU, Check 1 | 1.98 | 35.0 | 96.32 (96.24-96.62) | 1388 | 1.43 |
| 1024 | 128 | GPU, Check 4 | **1.39** | 35.0 | 96.32 (96.24-96.62) | 1068 | 1.00 |
| 1024 | 128 | GPU, Check 16 | 1.74 | 35.0 | 96.32 (96.24-96.62) | 1341 | 1.25 |
| 1024 | 512 | CPU 1 thread | 21.36 | 52.3 | 95.42 (95.18-95.53) | 23970 | 25.23 |
| 1024 | 512 | CPU 4 threads | 7.57 | 52.3 | 95.42 (95.18-95.53) | 8994 | 8.94 |
| 1024 | 512 | GPU, Check 1 | 1.01 | 52.2 | 95.42 (95.18-95.52) | 1836 | 1.20 |
| 1024 | 512 | GPU, Check 4 | **0.85** | 52.2 | 95.42 (95.18-95.52) | 1529 | 1.00 |
| 1024 | 512 | GPU, Check 16 | 0.94 | 52.2 | 95.42 (95.18-95.52) | 2047 | 1.11 |

The crossover, between the grid's cells (`make gpulearn OLANG=olang ARGS="3 3 128 64 1,4,gpu4"` and `ARGS="3 3 256,512
32,64 1,4,gpu4"`, 3 rounds):

| hidden | batch | batch x hidden | CPU 1 thread | CPU 4 threads | GPU, Check 4 | faster |
|---|---|---|---|---|---|---|
| 128 | 64 | 8192 | **1.39** | 1.69 | 2.10 | CPU 1.51x |
| 256 | 32 | 8192 | **2.08** | 2.56 | 3.51 | CPU 1.69x |
| 256 | 64 | 16384 | 2.24 | 2.53 | **2.00** | GPU 1.12x |
| 512 | 32 | 16384 | 4.32 | 4.91 | **3.73** | GPU 1.16x |
| 512 | 64 | 32768 | 4.70 | 4.91 | **2.19** | GPU 2.15x |

Accuracy, 15 seeds at batch 32 (`make gpulearn OLANG=olang ARGS="15 3 128,1024 32 1,gpu4"`): hidden 128 CPU 97.12%
(96.47-97.39), GPU 97.12% (96.47-97.30), 1.28 against 3.90 s an epoch; hidden 1024 CPU 96.11% (94.99-96.56), GPU 96.04%
(94.78-96.74), 11.53 against 4.07 s.

**After the rebase onto f151566** (master's NudgedRate per row in two norms, and gemm splitting a product over tasks
only per 16M multiply-adds; measured on the rebased tree, `make gpulearn OLANG=olang ARGS="3 3 128,256,512 32,64,128
1,4,gpu4"`, s an epoch, CPU 1 thread / 4 threads / GPU Check 4): every backend is 8-12% slower an epoch (the host's
certificate each lesson), and four threads now equal one thread below batch x hidden 16384 (the split no longer pays
for what it costs there). The crossover stays at 16384, narrower:

| hidden x batch | 128 x 32 | 128 x 64 | 256 x 32 | 128 x 128 | 256 x 64 | 512 x 32 | 256 x 128 | 512 x 64 | 512 x 128 |
|---|---|---|---|---|---|---|---|---|---|
| batch x hidden | 4096 | 8192 | 8192 | 16384 | 16384 | 16384 | 32768 | 32768 | 65536 |
| CPU 1 / 4 threads | 1.42 / 1.40 | 1.63 / 1.53 | 2.25 / 2.23 | 1.68 / 1.63 | 2.35 / 2.35 | 4.52 / 4.58 | 2.53 / 2.57 | 4.91 / 4.84 | 5.18 / 4.41 |
| GPU, Check 4 | 4.10 | 2.42 | 3.88 | 1.40 | 2.26 | 4.16 | 1.41 | 2.43 | 1.38 |
| faster | CPU 2.92x | CPU 1.58x | CPU 1.74x | GPU 1.16x | GPU 1.04x | GPU 1.09x | GPU 1.80x | GPU 1.99x | GPU 3.19x |

The sweeps a lesson are unchanged; the accuracies the same within the seeds (CPU and GPU still equal at batch 128).

Reproduce: `make data OLANG=olang` once, then the commands above; one run alone is
`make mnistlearn OLANG=olang ARGS="3 60000 1024 128 1 0.001 0.5 gpu"` (`gpu1`, `gpu16`, a thread count, or `auto`).

## The GPU route, GP6: ad on the GPU, G-M4 (wt-gp6, 2026-11-18)

docs/redesign.md 1.10 and 4.4's GP6 (the as-built note is in 1.10). `Graph.Plan(r, on = Gpu)`: `ad/mirror.olang` (the
arena as one device buffer at the CPU plan's offsets; a graph's node lowered to a `Step`, run by cuBLAS and
`kernels/gpu/src/ad.cu`), `kern.Kit` (many kernels of one text, one module), `optim.GpuAdamW` (AdamW and the clip on the
device), `ad/gpu_test.olang` (in `make gputest`), `bench/gpuad.olang` (`make gpuad`).

**Tests** (`make gputest`: 42 passed, GP4's 34 and GP6's 8 in `ad/gpu_test.olang`): every activation and its gradient
(written, added, from x and from y) against `kernels/elem` at odd sizes, within 2e-6 of the largest value; softmax, causal
softmax, their gradients and log-sum-exp at rows of 1 to 1000; layer and RMS norms (stats, apply, dx, dgain, dbias); AdamW
bit for bit with nn/optim's over 7 steps with a decay mask; the perceptron (a full batch, 11, 1) and the character
transformer (the two variants without Rope, Dropout or BF16, and the character configuration at batch 8) forward and
backward against the CPU's (loss, logits, every parameter's gradient: 2e-4 and 5e-4 of the largest) and a second GPU run
identical bit for bit; 20 AdamW steps clipped to norm 1 with the optimizer on the device against the host's (loss within
1e-2 and the norm within 2e-2 of the CPU's at every step: 20 steps of rounding drift, and it learns); what has no GPU form
refused at Plan by name. The step's allocation is pinned (`assert not ... .Allocates` on Forward, Backward, a launch, a
product, attention, AdamW's step).

**G-M4** (`make gpuad OLANG=olang ARGS="5 100"`, 2026-11-18; MNIST from `make data`; Ryzen 5 7600, RTX 4060 under WSL2,
rounds interleaved CPU 1 thread, CPU 4 threads, GPU, medians of 5):

| | CPU 1 thread | CPU 4 threads | GPU | GPU against the best CPU |
|---|---|---|---|---|
| M1: an epoch of 60000, 784-128-10, batch 128, AdamW (training only) | 0.276 s | 0.275 s | 0.136 s | 2.0x |
| M2: a step of the character transformer, 16 x 64 (forward, backward, clip, AdamW, loss read) | 59.61 ms | 43.46 ms | 5.62 ms | 7.7x (10.6x one thread) |

The same epoch loss and test accuracy on all three (0.4156, 93.53%). The M2 step is 182,000 characters/s on the GPU against
17,000 (one thread). The perceptron's step is small (a batch of 128 through 100,000 parameters: 0.136 s over 469 batches is 0.29 ms a
batch, mostly launches and the loss's wait), so its gain is 2x; the transformer's is the products and attention, 7.7x.
**Verdict for G-M4**: `ad` on the GPU pays for the transformer at once, and the perceptron by 2x - the element kernels
and attention as built are enough; the next lever is fusion (bias, activation and residual into the products' epilogues;
CUDA graphs for the step's ~250 launches), GP7's, not more kernels.

Reproduce: `make data OLANG=olang` once, then `make gpuad OLANG=olang ARGS="5 100"` (`mlp` or `lm` as a third argument runs
one half). The GPU tests: `make gputest OLANG=olang`.

**Limits.** F32 and F64 graphs only (BF16 graphs refused); Dropout, Rope, convolutions, scans, permuted/sliced views and a
MatMul of two batched operands stay on the CPU; `nn/train`'s loops on a GPU graph work through Value/Params (a transfer a
step) - a loop wanting the speed steps `optim.GpuAdamW` itself, as bench/gpuad does. Attention's causal products are full
(cuBLASLt has no triangular product): half the scores' work is wasted. A short batch runs attention for every sequence.

## The GPU route, GP7: what the measures ask for (wt-gp7, 2026-11-18)

docs/redesign.md 1.10 and 4.4's GP7: each lever kept only where it measured a gain. A compiler build ran beside these
measurements (load average 1.5 to 15), so every comparison below is the same binary or two binaries run back to back, and
the figures are the medians of the tool's own rounds; read ratios, not absolutes.

**Kept**

1. **ProductI8's dequantization is one pass** (`oann_dequant_i32` in `src/elem.cu`, launched by `ProductI8.Dequant`; the
   second F32 product, its plan and `cublasSdgmm` are gone). `make gpult OLANG=olang ARGS=5`, "GPU I32 and F32 out", us:
   2048x2048x2048 2466 -> 314 (the I32 product alone 200); 64x768x3072 -> 18.3; 128x784x128 -> 13.4. out equals scale[j] c
   exactly (the test's bound unchanged, 2e-6 relative to F64's). G-M5's blocker is gone.
2. **The settle's sweeps are CUDA graphs** (`gdev.Graphs`, `Mirror.sweeps`): each group of `Check` sweeps is captured the
   first time it is asked for and replayed after; the key is the group, the rows, the budget's bits and the populations'
   structure, 256 graphs at most (more run as launches). Same binary, `OANN_GRAPHS=0` against 1 (`make gpusettle`, medians of
   5 rounds; us a cold settle / a lesson at batch 32, 128, 512): 560 / 1312, 560 / 1627, 720 / 3021 without; 372 / 1104,
   412 / 1323, 575 / 2704 with (about 1.3-1.5x a settle, 1.1-1.2x a lesson). On MNIST (`make gpulearn OLANG=olang ARGS="5 1
   128 128,512 1,gpu4,gpu8"`, hidden 128, the same accuracy and sweeps): an epoch at batch 128 2.13 s -> 1.06 s (the CPU one
   thread 2.07), at batch 512 0.50 -> 0.41 s. The crossover moves: batch 64 (8192) is now a tie with the CPU (1.71 s with
   `Check` 8, 2.10 with 4, against 1.68) where GP5 had the CPU 1.5-1.7x ahead; batch 512 x 32 (16384) the GPU is 1.7x ahead
   (GP5: 1.1x); batch 32 x 128 (4096) the CPU is still 2.3x ahead. `SettleOn` stays at 16384 (8192 is a tie, not a win).
   `Check` 4 stays the default (8 was 0.96 against 1.06 s at 128 x 128 and slower at 64 x 512: within noise).
3. **ad's whole step is one graph** (`GpuAdamW.Train(loss, n, clip)`: inputs up, then Forward, Backward, clip and AdamW
   captured at the second call of a (loss, n, clip) and replayed; the AdamW kernel reads its three step scalars from a
   device array the host fills, so a replay has each step's). Bit for bit with the launches one by one (a new test over 12
   steps of changing batch size). `make gpuad`, M2 transformer step: 5.63 -> 5.02 ms (11%) before the next item; 4.58 ->
   3.71 ms (19%) after. M1: 0.164 -> 0.152 s an epoch in the first (quiet) run, within noise when loaded.
4. **Column sums tiled** (the profile below): the bias gradient and the norms' gain/bias gradients took one thread a
   column over 1024 rows (35-75 us a node); a block of 1024 threads, 32 columns by 32 row lanes, partials added in lane
   order (fixed, no atomics) brings Norm backward from 656 to 157 us and Linear backward from 1659 to 1032 us a step
   (`make gpuad ARGS="1 1 prof"`, `Graph.GpuProfile`). M2 eager 5.63 -> 4.58 ms.

**Skipped, with numbers**

- **Fewer launches in the sweep**: already done in GP4 (the constants' copies and the residual word's zeroing are in the step
  kernel and one memset a settle; 4 launches a sweep). Nothing left to fold.
- **Capturing `prepare`** (the closed form and the folded constants) as a graph too: 385/416/558 against 372/412/581 us a
  cold settle: no change; removed.
- **Fused attention** (scores, softmax and the product in one launch a row; dq, dk and dv in two): written and timed (not
  correctness-tested), and slower than the batched products in the graph step (4.01 against 3.76 ms in an interleaved A/B), so removed.
  The profile says attention is 1.0 ms of the step's 3.3 (fwd 338, bwd 644 us); a fused kernel would need tiling in shared
  memory to beat cuBLAS's, not a row a block.
- **Bias and activation into the products' epilogues**: Linear's forward is 17 us a node of which the bias launch is a few;
  with the step a graph the saving is at most about 2% (22 nodes); not done.

**Tests**: `make test` 364 passed, 0 failed; `make gputest` 43 passed (GP6's 42 plus `graphed` (Train against the launches one by one, bit for bit) and the
AdamW kernel test reading its scalars from the device, and the norm test's tiled launch).

Reproduce: `make gpult OLANG=olang ARGS=5`; `make gpusettle OLANG=olang` and `OANN_GRAPHS=0 ./build/bench_gpusettle` for the
A/B; `make gpulearn OLANG=olang ARGS="5 1 128 128,512 1,gpu4,gpu8"`; `make gpuad OLANG=olang ARGS="5 100"` (`prof` as the
third argument prints the step's time on the device by kind, `gpulm` runs the GPU's two transformer rows only).

## The GPU route, GP8: views, stacks, and oagent's e2 on the GPU (wt-gp8, 2026-11-18)

What oagent's experiment graph `experiments/e2-ebr/ebr.olang` (an energy-based reasoner on Sudoku: token-axis mixing by
`g.Permute(h, 0, 2, 1)`, a hand-written energy gradient, deep supervision) needed to plan on the GPU, and what it gains.
Trying it found Permute refused first; after views, two GP6 bugs that only a graph like e2's reaches.

**On the GPU now** (`ad/mirror.olang`, `ad/graph.olang`, `kernels/gpu/src/ad.cu`):
- **Views.** Permute, Slice and Expand plan on the GPU. A view still copies nothing; what reads it through a recorded
  `Contiguous` (Linear, Add, Mul, the element operations, Reshape) gets a strided gather (`oann_ad_gather`: up to rank 6,
  the view's dims and strides in a geometry table, `Step.Geo`), and a view's gradient goes back into its base's as the
  CPU's `viewBack` does: Permute's and Slice's as the same gather with the strides inverted (Slice's with a window,
  zero outside), Expand's by `oann_ad_unexpand`, a sum over the broadcast axes in dy's order (no atomics, deterministic,
  the CPU's order of adds). A product by a matrix reading a view through leading strides of its own (no copy recorded
  for it on the CPU) is refused by name ("MatMul of a strided view").
- **MatMul of two stacks** (both operands batched, or neither: rank 3, or rank 4 with two leading axes): cuBLASLt's
  batched products (`MatPlan`: forward and both gradients planned at Plan for the planned batch, a short batch running
  every matrix as attention does), each operand read through its own strides - attention written out by hand, heads as
  permuted views, with no copy.
- **Fixed on the way (GP6's)**: (1) a `[batch]` vector (rank 1, the batch its only axis) ran on 0 elements
  (`Rows / batch * n` with Rows 1): e2's per-puzzle convergence loss and its gradient were zero on the GPU -
  `Step.BatchLast`, one row of the entries in use; (2) an input written between Forward and Backward (e2 sets the
  convergence loss's targets from the decoded output) never went up - an input handed out by `Data`, `Ints` or `Value`
  is marked and goes up before the Backward; (3) uploads went on the copy stream, ordered before later launches but not
  after earlier ones, so a Forward still in flight could read the next input (a flaky test showed it) - ad's uploads now
  go on the compute stream. And `Value`/`Grad` on a GPU graph bring down that node's storage alone (a generation stamp a
  node), not the whole activation region as before: e2's stepper reads one [B, 81, 32] state per step.

**Tests** (`make gputest`: `ad/gpu_test.olang` 11 passed - GP7's 9 plus): `viewsCase` - Permute (rank 3 and 4, of a
parameter, of a view), Slice (a middle axis, the last axis, of a view), Expand (of a slice; of a parameter, then
permuted), a loss over `[batch]` vectors whose weights are set after the Forward: loss, logits and every parameter's
gradient within 2e-4 of the largest against the CPU (GP6's perceptron bound) at batch 5, 3, 1, 5, a second GPU run
identical bit for bit; `stacksCase` - attention by hand (heads as permuted views, rank-4 stacks, `tb`), a transposed
rank-3 product (`ta`), at batch 4, 3, 4, same bounds; `refused` adds the strided-view product. e2's own graphs, one
step from the same inputs (a `cmp=1` mode in a scratch copy): logits, the end state and q within 5e-7 relative,
every parameter's gradient within 1.1e-6 (ebr) and 5.6e-7 (loop) of the largest. `make test`: 373 passed, 0 failed.

**e2 train step, CPU against GPU** (2026-11-18; Ryzen 5 7600, RTX 4060 under WSL2; the experiment as it is - host
AdamW, the stepper's state read back every step, noise drawn on the host - with only a `gpu=1` switch and a step timer;
d 32, k 81, m 64, 2 blocks, H 4, n 6..8; ms a step after a warm-up, seed 1; the machine shared with other sessions'
runs, load average 4.7-7.3 on 6 cores, so the CPU runs at 1 and 2 threads only; read ratios):

| arm, batch | CPU 1 thread | CPU 2 threads | GPU | GPU against the best CPU | of which host noise (CPU / GPU) | the rest: CPU / GPU |
|---|---|---|---|---|---|---|
| ebr, 32 | 156.2 | 149.1 | 26.9 | 5.5x | 12.3 / 13.5 | 136.8 / 13.4 (10x) |
| loop, 32 | 119.6 | 117.9 | 22.8 | 5.2x | 12.4 / 13.9 | 105.5 / 8.9 (12x) |
| ebr, 128 | 605.0 | 596.1 | 77.7 | 7.7x | 47.1 / 49.5 | 549 / 28.2 (19x) |
| loop, 128 | 467.1 | 469.4 | 73.0 | 6.4x | 48.8 / 53.9 | 418 / 19.1 (22x) |
| ebr, 512 | 3097 | 3158 | 303.5 | 10.2x | 240 / 201 | 2857 / 103 (28x) |
| loop, 512 | 2346 | 2334 | 344.2 | 6.8x | 247 / 235 | 2099 / 110 (19x) |

The same trajectory on both (step 50's cross entropy and the run's ends agree; the graphs within rounding, above).
The CPU's second thread buys nothing: e2's products (81 x 32 x 81 a puzzle) are too small to split. **On the GPU the
experiment's own host work is now most of a step**: the noise (`r.Normal` per element, B x 81 x 32 a stepper step and 4
of them a trainer step) is half the step at batch 32 and two thirds at 512; the GPU's part (stepper, trainer forward and
backward, the host AdamW's transfers) is 13 ms at 32 and 103 ms at 512. Next levers, the experiment's: draw the noise on
the device (or reuse one draw scaled per slot), step `optim.GpuAdamW` (or `Train`, one CUDA graph) instead of the host
AdamW and its parameter copies between the two graphs. **The ep arm** (oann's `ad/descent`, which writes the state and
reads dE/dx and dE/dtheta through the host every settle step) works on a GPU graph unchanged: 50 steps at batch 32,
265 passes a step, 0.2 min on the GPU against 1.9 min on the CPU at 2 threads (the same cross entropy 2.0900, the same
settle steps) - about 0.9 ms a pass, bound by the per-pass round trip (state up, gradients and parameters down);
a settle kept on the device (the descent step as kernels) would be the lever, not done here.

Reproduce: `make gputest OLANG=olang`; e2: copy `ebr.olang`, add `gpu=1` (`device.SetDefault(device.Backend.Gpu)` before
the graphs are planned), build with `-define Gpu=true`, `arm=ebr|loop batch=32|128|512 steps=100 evalcount=1`.

**Still refused on the GPU**: BF16/F16 graphs, eager and decoding graphs, Dropout, Rope, Conv2d, MaxPool2d, Scan,
SelectiveScan, DeltaRule, Custom, Sqrt, Log, Shift, StablemaxCE, a product by a matrix reading a view through
leading strides, and a product of a batched stack by an unbatched one.

## One graph energy, three trainers (wt-bothmodes, 2026-11-18)

ad/descent: a state settled by gradient descent on an energy recorded on the graph (dE/dx by its Backward), the
parameters' gradient of a cost at the fixed point by backpropagation through the last h steps, the exact implicit
gradient (conjugate gradients on the adjoint) or equilibrium propagation (free and nudged settles, dE/dtheta's
contrast - first derivatives only). Second derivatives are central differences of the first (the graph records no
second-order backward). Tested on a quadratic and a quartic convex energy: implicit against central differences of the
settled cost within 1e-6, Unrolled(400) against implicit within 1e-6 (Unrolled(3) off by more than 1e-3), centered EP
(beta 1e-3) within 1e-5, one-sided within 1e-2; 30 EP steps halve the settled cost. Found on the way: two Backwards
after one Forward must run the later-recorded loss first (the plan's liveness assumes it; ad/README.md).

## Anderson's acceleration for the exact adjoint (wt-anderson, 2026-11-18)

dyn/learn's exact implicit gradient (`Implicit.Exact`) now solves its adjoint with Anderson's method where the system's
scheme asks for it: one setting, `System.Use(settle.Sweeping(Anderson = m))`, accelerates both the forward settles and
the adjoint, at the same depth, on the settle's history (no new room: Reserve's already covers one phase's rows;
ImplicitGrad still allocates nothing). The adjoint's safeguard is looser than a settle's - the history is dropped only
where the residual grows tenfold: with the settle's (drop at any rise) the adjoint took *more* sweeps than plain
(67 against 45 on 784-256-10), since a linear iteration's sup-norm residual under Anderson is not monotone; with no
safeguard and with the tenfold one the sweeps were the same. The forward settle's safeguard is unchanged. Truncated
gradients (TruncatedGrad) stay plain: their k steps are the Neumann series by definition.

`make adjointbench ARGS="5 32 1e-6"` (bench/adjoint.olang): tanh rate systems, symmetric connections, random inputs,
batch 32, adjoint tolerance 1e-6, F64, the median of 5, one thread:

| system | kappa | plain sweeps | ms | Anderson(3) sweeps | ms | Anderson(5) sweeps | ms | gradient vs plain |
|---|---|---|---|---|---|---|---|---|
| 784-256-10 | 0.9 | 45 | 4.54 | 27 | 4.95 | 22 | 4.81 | 1.2e-06 |
| 784-256-10 | 0.99 | 67 | 6.31 | 36 | 6.73 | 29 | 6.41 | 2.8e-06 |
| 784-256-256-10 | 0.9 | 57 | 30.01 | 34 | 26.97 | 27 | 22.31 | 7.0e-07 |
| 784-256-256-10 | 0.99 | 85 | 47.05 | 48 | 32.40 | 35 | 25.99 | 1.2e-06 |

Reading: Anderson(5) halves the adjoint's sweeps or better (2.0-2.4x fewer), Anderson(3) 1.7-1.9x. In time it pays
where a sweep is dear: the 3-layer system is 1.35-1.8x faster at depth 5; the 2-layer one, whose sweep is one small
product, is level (the history's dot products cost about what the sweep saved). The gradients agree with the plain
adjoint's to the tolerance. The test (dyn/learn, kappa 0.97, both wirings): plain 137 / 44 sweeps, Anderson(5) 47 / 27,
gradients within 1e-9 at tolerance 1e-13. examples/mnist_learn has no exact-gradient path (EP lessons only), so it is
not measured here; a 3-layer system that needs Anderson on the forward settle (oagent e6) now gets it on the adjoint
from the same `Use`.

## Next

- The open work of docs/redesign.md: the GPU route (question 1) - GP1 to GP5 are done (above; G-M2's verdict: the
  settle on the GPU from batch x hidden 16384, `Check` 4); GP6 (`ad` on the GPU, above) is done. For GP7 G-M2 says the GPU's
  fixed cost a settle (about 1 ms whatever its size) sets the crossover: CUDA graphs or fewer readbacks would move it
  below 16384.
- INT8 on the GPU: a one-pass kernel dequantizing GemmI8's I32 product (kern.olang) instead of the second, F32-summed
  product (GP3: 192 against 2466 us at 2048^3); then G-M5.
- The nudged pair's certificate per batch: `NudgedRate` with the block norm certifies 60% of MNIST's rows at epoch 3
  (44% before; above), no batch of 32. Next lever: a budget per row (a per-row stop in dyn/settle), or a tighter
  hidden-to-readout block.
- Compacting retired units (above, if it pays).
- INT8: shallow products (the CNN's depth 9 and 144).

## Done since C3

- The agent rebuilt on `dyn` (`DeltaWrite`, `ThreeFactor`, `Homeostasis`) and oagent's `agent/legacy/` dropped: oagent
  main 4ce216c, every example matching legacy within noise.
- Pruning at run time beyond the masks (docs/redesign.md question 3): above.

## History moved out of the READMEs (2026-11-16)

- C3 removed `board.olang`, `circuit.olang`, `linalg`, `quant`'s product, `datasets/` (the first four of `data/`'s modules were
  `datasets/`'s) and `bench/sparse_kernels.olang`; `git show pre-c3:PATH` has any of them. The parity tests of `board/engine` against
  `board.olang` (same image, settles, lesson, golden vectors; XOR 400 lessons, spiking XOR 40) and the golden digests' equality with
  board.olang's dumps were checked before C3. MNIST through the board was first taught in circuit: 90.05 / 91.02 / 91.09%.
- Speeds against linalg's product, `Gemv` on a settle's blocks, `GemmI8` against quant's product, dense against sparse kernels
  (M4, M5, M10, M12) and the settle against `circuit.olang`'s (M7) were measured before C3; drivers at branch `pre-c3`.
- The settling agent's programs (`bandit_settle`, `sleep_retention`, `nights`) moved to oagent with the agent at C3.
- The repro cases `lambdalend` and `resultgrowth` (on the old `linalg`) were removed with it.

## Measured on the data (2026-11-16)

`make data` fetched MNIST into `data/mnist` (storage.googleapis.com mirror: train-images 47040016 bytes = 16 + 60000 x 784,
train-labels 60008, t10k-images 7840016, t10k-labels 10008 = 10000 labels; pixel mean 0.13066, sd 0.30811 against MNIST's
0.1307, 0.3081). Tiny Shakespeare came with `make charlm` (1115394 characters, vocabulary 65). One core, olang from ~/.local.
Reproduce each with the `make` target and ARGS shown (olang's `OLANG=olang`).

- **MLP** (`make mnist ARGS=3`, AdamW, batches of 128): test accuracy 93.53 / 95.46 / 96.30% after epochs 1-3, 0.43 s an epoch.
- **CNN** (`make cnn ARGS=2`, 20490 parameters): 97.29 / 97.93%, 18.2 s an epoch, resident memory flat (110 MB).
- **INT8** (`make int8 ARGS="mlp 3"`): F32 96.30%, INT8 dynamic 96.31%, static 96.29%; parameters 3.94x smaller; batch 1 2.38x / 2.77x
  (dynamic / static) faster than F32, batch 64 0.59x / 0.66x, batch 128 0.52x / 0.55x (slower, the known packed-kernel gap).
  CNN (`ARGS="cnn 1"`): F32 97.29%, INT8 97.33 / 97.30%, 3.88x smaller, 0.61-0.84x (slower at every batch).
- **Character transformer** (`make charlm ARGS=300`, 4 layers x 128, 809856 parameters, context 64, batch 16 x 64): validation loss
  4.194 at step 0, 2.360 at step 300, 80.7 ms a step (12683 tokens a second); the sample is still near-nonsense (300 steps).
  **charlm_int8** (`make int8lm`, on that checkpoint): validation loss F32 2.3598, INT8 2.3601; weights 3.91x smaller; a forward over
  1024 characters F32 28.96 ms, INT8 43.56 ms (0.66x); cached decoding 5021 characters a second F32, 8244 INT8 (1.64x); greedy
  text identical for 64 of 64 characters.
- **Settling `mnist_learn`, dense** (`make mnistlearn ARGS="3 60000 128 32 SEED"`, Adam 1e-3, beta 0.5, not restrained beyond
  `Certify(0.9)`): seed 1 95.72 / 96.50 / 96.71%, seed 2 (epoch 3) 97.10%, about 2.5-3 s an epoch, 30-32 sweeps a lesson. Matches
  docs/settling.md's unrestrained 95.69 / 96.48 / 96.58%.

### Pruning `mnist_learn` with `Rule.Prune`

`mnist_learn` takes four more arguments: the fraction kept (9th, default 1 = off), the pace (0.1), the lessons between ticks (200)
and the epochs relearned (1). A `Rule.Prune(Score.Magnitude, keep, pace, no regrowth, Schedule.Cubic)` (the default schedule) is
attached; the system trains 2 dense epochs, then each tick prunes and teaches 200 lessons until the fraction is reached, then 1 epoch
relearns, then `Sparsify(0.5)`. Reproduce: `make mnistlearn ARGS="2 60000 128 32 SEED 0.001 0.5 1 KEEP 0.1 200 1"`. Test accuracy,
and multiply-adds (`Macs`) of one settle at the sweeps an answer took (once = the pixels' folded constant; a sweep = the recurrent part):

| run | seed | accuracy | weights live | Macs once / a sweep | Macs a settle |
|---|---|---|---|---|---|
| dense, 3 epochs | 1 / 2 | 96.71 / 97.10% | 101632 | 100352 / 2560 | 138752 / 136192 |
| 50% kept (after Sparsify) | 1 / 2 | 97.34 / 97.33% | 50816 | 50141 / 2560 | 88541 / 91083 |
| 20% kept (after Sparsify) | 1 / 2 | 97.43 / 97.52% | 20327 | 20013 / 628 | 31945 (seed 1) / 33002 |

Pruning at 20% kept holds the dense accuracy (against a dense run given the same lessons, see below) and costs 4.3x fewer multiply-adds a settle (a sweep 628 against 2560 multiply-adds);
answers take more sweeps (18.7-19.9 against 14-15). At 50% the recurrent hidden-readout block is not thinned at all (2560 a sweep);
the pixel connection takes the cut. Two seeds only.

Fair dense control (wt-oannconv): the pruned runs teach 3 epochs plus 8 ticks of 200 lessons = 1600 more (none refused). A dense run
given the same lessons (`mnist_learn`'s 13th argument: `3 60000 128 32 SEED 0.001 0.5 1 1 0.1 200 1 1600`):

| run | seed 1 / 2 accuracy |
|---|---|
| dense, 3 epochs | 96.71 / 97.10% |
| dense, 3 epochs + 1600 lessons (same lessons as pruned) | 97.32 / 97.43% |
| 20% kept (after Sparsify) | 97.43 / 97.52% |

So most of the earlier gain was the extra lessons. Pruned to 20% is level with the equal-lesson dense run (+0.11 / +0.09 points, inside what two
seeds can show) with 4.3x fewer multiply-adds a settle: pruning costs no accuracy here, it does not add any measurable.

## Ahead and the settle options under the spectral default (wt-oannahead, 2026-11-17)

**`Ahead`.** The next batch is filled on a task while the step trains: `train.EpochAhead` (a classifier; `train.Ahead.For(g, c)`
holds the staging buffers) and `Config.Overlap` in `Fit` (`train.LmStepAhead`, `train.LmAhead.For(g, m)`); the task writes the
stage, the step reads the graph's input, the stage is copied over after the `join`. `Loader.Rows(b)` gives a batch's size.
`examples/mnist_mlp` and `charlm` take the word `ahead`. Tests (nn/train): the same losses and parameters as `Epoch` / `LmStep`
bit for bit over 100 epochs / 130 steps, and resident memory flat after the tasks settle (60 epochs). A failing batch source
is `TrainError.Draw`. Measured on a loaded machine (load 10-15 on 4 cores), alternated runs, one thread:
`mnist_mlp 3 adamw 1 1`, mean epoch plain / ahead: 0.389 / 0.410, 0.381 / 0.393, 0.354 / 0.439 s (losses identical); `charlm 300 1 1`,
ms a step plain / ahead: 93.1 / 93.1, 94.2 / 87.3, 92.5 / 91.6 (a step is 90 ms, a batch's fill microseconds). Nothing gained:
MNIST's fill is 8% of an epoch and the task competes for a core under load (as in design-history.md); a language model's is nothing.
Worth turning on only with a core to spare and a dear fill. olang: a join keeps ~110 bytes (repro/spawnleak.olang; joinleak.olang's
case now reads 2 B a join, this one 113 B), and the tests allow for it.

**Settle options under `Norm.Spectral` with the Euclidean stop test.** `mnist_learn 3 60000 128 32 SEED 0.001 0.5 1 1 0.1 200 1 0 WAY
spectral` (dense, seeds 1 / 2, epoch 3) and `xor_settle 300 8 1 0.05 0.5 WAY spectral` (3 runs), run in turn on a loaded machine
(load 10-15), so times carry the load; sweeps, refused and accuracy do not:

| MNIST dense | sweeps a lesson | sweeps an answer | us a lesson | us a settle | refused (epochs 1-3) | accuracy |
|---|---|---|---|---|---|---|
| plain | 31.8 / 31.9 | 10.6 / 10.9 | 1694 / 1767 | 244 / 295 | 0 / 0 | 96.98 / 96.90% |
| Gauss-Seidel | 23.2 / 23.5 | 5.7 / 5.8 | 1603 / 1436 | 211 / 270 | 0 / 0 | 96.90 / 96.89% |
| Anderson 5 | 21.6 / 22.4 | 8.8 / 8.9 | 2094 / 2282 | 394 / 546 | 1 / 2 | 96.83 / 96.82% |
| both | 16.3 / 16.9 | 4.9 / 5.0 | 1890 / 1859 | 321 / 331 | 1 / 2 | 96.86 / 96.83% |

| XOR (8 hidden) | sweeps a lesson | us a lesson (3 runs) | refused | loss at 300, right |
|---|---|---|---|---|
| plain | 8.5 | 14.6-16.1 | 0 | 0.0022, 4/4 |
| Gauss-Seidel | 6.5 | 16.9-30.7 | 0 | 0.0021, 4/4 |
| Anderson 5 | 7.0 | 13.6-13.8 | 0 | 0.0022, 4/4 |
| both | 5.5 | 15.0-17.6 | 0 | 0.0021, 4/4 |

Reading: under the spectral certificate the system sits at kappa 0.9 throughout (gain 6.8-7.6), so plain sweeps already settle in
32 a lesson (against 42 last round's sup-tolerance run), nothing is refused with plain or Gauss-Seidel, and the options' effects are
as before: Gauss-Seidel saves 27% of a lesson's sweeps and half an answer's, is as fast or faster on MNIST, and level or slower on
XOR (its wall time is noise-dominated there); it costs 0.01-0.08 points. Anderson saves 30% of a lesson's sweeps but each costs more
(a settle 1.6-1.9x slower), and refuses 1-2 lessons in epoch 1 (it no longer stalls: the certified system is within its safeguard).
Both together: fewest sweeps (16 a lesson, 5 an answer) but slower than plain per lesson. **The default stays plain Jacobi**: no
option is faster on every case with unchanged accuracy (Gauss-Seidel is the near miss: -0.08 points on seed 1, XOR time level or worse).
Logs in the worktree's logs/ (not committed).

**INT8 weights at batch 1: two rows per pass (`dotRow2`).** The BF16 fix's one-row `dotRow` cost INT8-under-F32 streaming 10-20%. `dotRow2` takes two rows with each row's eight chains and order unchanged (the storage test's bit-for-bit sums hold) and reads the vector once; the INT8 storage type only (`pairedRows`). For F32 and BF16 the same loop was 0.4-0.7x F32 (LLVM's SLP vectorizer again), so they keep `dotRow`. bench/storage (loaded machine, alternated 3x, bench shapes 1x512x512 and 1x4096x1024 added), INT8 vs F32: 1x784x128 0.69-0.94x -> 0.83-0.90x (5.0-5.2 us against 6.9-8.4), 1x1024x1024 2.58-2.77x -> 2.72-3.17x, 1x4096x1024 2.39-2.47x -> 2.70-2.81x, 1x512x512 level; BF16 and F32 unchanged.

## Products split under one join (wt-gemmjoin, 2026-11-21)

A spawn and its join cost about 30 us a task on the 4-core VM, most of it the sleeping worker's wake (a plain C condvar
round trip measures 25 us there; a spinning one 0.6 us). kernels/gemm split a large product once per packed block of
op(B) - a join per 256 of the depth, four for a depth of 784 - so a split paid that four times. Now `packedProduct` starts
its tasks once: each runs the whole blocked loop over its own part, a run of tile rows (where the result is at least as
tall as wide; every task packs op(B), the narrower panel) or of tile columns (every task packs op(A)), with panels of
its own from the workspace (Reserve's room already holds one per thread). The split threshold drops from 16 M to 4 M
multiply-adds a task (2 M, and at most a task per three tiles, measured level). Sums are unchanged: the tests (a
column-split product past a block of columns, and an epilogue on one, added) hold bit for bit across thread counts.

bench/storage F32, 4 threads, alternated runs (us; one thread unchanged within noise):

| shape | before | after | one thread |
|---|---|---|---|
| 64 x 784 x 128 | 130 | 122-133 | 130 |
| 128 x 784 x 128 | 218-227 | 155-179 | 221 |
| 1024 x 512 x 128 | 608-643 | 408-556 | 1050 |
| 1024 x 128 x 512 | 369-553 | 358-579 | 900 |
| 512^3 | 887-1065 | 660-920 | 1820 |
| 1024^3 | 4810-5050 | 4750-5230 | 14400 |

bench/lm (the character transformer, 20 steps), 4 threads: a step 63.1 / 63.5 -> 53.8 / 57.0 ms, its products 42.8-48.2 ->
33.7-34.5 ms. G-M1's 4-thread slowdown at 32 and 128 rows (fixed 2026-11-17 by not splitting them) now turns into a
gain at 128 rows. Left: the workers' wake itself (an olang runtime matter: spinning briefly before parking would bring a
split to a few microseconds), and duplicate packing of the narrower operand, which caps 64-128-row products.

Also re-checked repro/ against olang 3b47ad852: every bug but `tempref` is fixed (repro/README.md).

## Halting on a stable answer (wt-stable, 2026-11-21)

`dyn/halt.Stable` is a general stop rule for any loop that refines an answer step by step: a row holds once its answer
(one or more integers - a class per readout group, an index, a token) has been unchanged for k steps; allocation-free,
usable beside a residual-based stop. The settle takes it as `settle.Budget.Stable` (k > 0): after each sweep every row's
answer is read (each readout group's largest unit); with per-row stops (a Euclidean budget, Jacobi, no Anderson) a row
that holds stops at that sweep as a converged row does, and the settle ends once every row has stopped one way or the
other (`Settlement.Halted`; a rate system on the CPU - the GPU's settle refuses it by name). Tested on a chain whose
inputs are turned round after a first settle (dyn/system): a damped settle to 1e-12 takes 51 sweeps, halting at k = 3
takes 9 with all 130 answers the converged state's, and the turned rows stop at their own sweeps while the unturned
ones stop on the tolerance at once.

`examples/mnist_learn` takes a 17th argument, halt: after training, the test set is answered again for each k up to it.
Seed 1, 3 epochs, 784-128-10, batches of 32 (the test answers settle as they do in training, at the lesson tolerance):

| certificate | tolerance only | k = 1 | k = 2 | k = 3 | k = 4 | k = 5 |
|---|---|---|---|---|---|---|
| none (sup) | 96.71%, 14.7 sweeps | 95.86%, 2.0 | 96.56%, 3.8 | 96.57%, 4.8 | 96.64%, 6.0 | 96.65%, 7.0 |
| spectral (rows stop on their own) | 96.89%, 10.7 | 96.56%, 2.1 | 96.65%, 3.3 | 96.82%, 4.4 | 96.87%, 5.5 | 96.87%, 6.6 |

So an answer needs a third to a half of the sweeps the residual asks for at a cost of 0.02-0.14 points (k 3-5); one
seed. Reproduce: `make mnistlearn ARGS="3 60000 128 32 1 0.001 0.5 1 1 0.1 200 1 0 plain spectral 5"`.

## Message passing over sparse graphs, sum or mean (wt-message, 2026-11-21)

`nn/message`: `Edges` holds each node's incoming edges as compressed rows, refilled from an edge list per batch without
allocating; `Passing` is a `Graph<F32>` operation (through `Custom`) giving each node the sum (`Agg.Sum`) or the mean
(`Agg.Mean`) of its in-neighbours' rows, its backward the transpose; `Layer` is act(x Self^T + m Msg^T + B). A batch of
graphs is one graph of disjoint parts. Tested against the edge list directly, values and gradients, both aggregations.

`examples/graph_count` (`make graphcount`): nodes red or blue; two message-passing layers of width 16 and a dense head;
1500 steps of AdamW on batches of 256 nodes of small graphs (8-16 nodes, 1-4 in-edges a node), then 40 batches each of
graphs like them and of larger, denser ones (64-128 nodes, 1-10 in-edges). Node accuracy, seeds 1 / 2 / 3:

| task | aggregation | small graphs | larger graphs |
|---|---|---|---|
| at least three red in-neighbours | sum | 100 / 100 / 100% | 92.8 / 96.8 / 95.3% |
| | mean | 93.9 / 94.5 / 93.8% | 58.4 / 60.6 / 59.0% |
| more red than blue in-neighbours | sum | 100 / 100 / 100% | 99.98 / 100 / 98.9% |
| | mean | 100 / 100 / 100% | 95.1 / 94.9 / 94.8% |

A mean cannot count (it loses the degree), so it misses the counting task even on the graphs it was trained on and
falls to near chance on larger ones; a sum both counts and, from two colour columns, compares (red - blue), so it holds
up on graphs larger and denser than any it saw. A run takes half a second.

## A small self-attention encoder over variable-length histories (wt-history, 2026-11-21)

`nn/history.Encoder`: a projection of each entry to width D, a learned position, one causal pre-normalized
`AttentionBlock`, a final norm, and a pooling by a batched weights input (`Weights`: the mean over the valid positions,
or the last valid position). Histories are right-padded; causal attention means no valid position reads padding and
everything else is per position, so the encoding and every parameter gradient are bit for bit the same whatever the
padding holds (the test fills it twice), and one planned graph serves every length.

`examples/history_encode` (`make historyencode`): histories of 1-16 symbols of 8, one-hot, zeros past the end; the
encoder (D 32, 4 heads, 13.6-13.8k parameters) against a flat perceptron over the padded history (128-256-256-classes,
99-101k parameters); AdamW at 0.002, batches of 64, 3000 steps, then 50 fresh batches. Test accuracy, seeds 1 / 2 / 3:

| task | encoder | flat perceptron | flat, 15000 steps (equal time, seed 1) |
|---|---|---|---|
| most frequent symbol (encoder: mean) | 99.75 / 99.94 / 99.88% | 98.94 / 98.75 / 98.59% | 99.34% |
| last symbol seen before (encoder: last) | 99.41 / 99.78 / 97.03% | 87.06 / 87.31 / 87.62% | 97.78% |

At equal steps the encoder wins with a seventh of the parameters, most on the comparison of the last entry against
the rest (+10-12 points). A step costs 2.7-2.9 ms against 0.5-0.6 ms on the CPU (attention at Ctx 16 is dominated by
small products), and at equal time the perceptron closes most of the gap (97.8% against 99.4% on repeat). The encoder's
lead is in parameters and steps, not yet in CPU time at this size.

## Best-of-N by a model's own score (wt-bestof, 2026-11-21)

`nn/bestof.BestOf<T>`: N candidates per row (settles from different starts, samples, descents from noise), each scored
by the model itself - an energy or a loss (`Order.Lowest`), a likelihood or a verifier's score (`Order.Highest`) - and
the best kept per row (`Offer`, `OfferRows`, or `Search` with a function making the candidates); NaN never wins, the
first of equals is kept, nothing allocated.

`examples/best_of` (`make bestof`): a continuous Hopfield memory on dyn - 64 tanh units connected reciprocally by Hebbian
weights (gain 4) storing 4 random +-1 patterns, the cue a stored pattern with 20% of its signs flipped as a drive of
0.3. Each query is settled from 16 random starting states; candidate k is the signs of the k-th settle, scored by
`dyn/energy.Of`. Recall (every sign of the cued pattern) over 512 queries, seeds 1 / 2 / 3:

| tries | one settle | lowest energy of N | any of N (oracle) |
|---|---|---|---|
| 1 | 43.0 / 35.7 / 37.3% | | |
| 2 | | 63.5 / 52.3 / 60.5% | 63.5 / 59.4 / 60.7% |
| 4 | | 85.2 / 62.3 / 82.6% | 85.2 / 81.6 / 83.4% |
| 8 | | 96.5 / 64.8 / 95.7% | 96.5 / 95.7 / 97.1% |
| 16 | | 99.8 / 64.1 / 98.2% | 99.8 / 99.8 / 100.0% |

Where the cue's pattern is the deepest minimum the energy finds it as often as the oracle does (seeds 1 and 3); seed 2's
patterns hold a pair the energy confuses, and selection stalls at 64%. A weaker drive (0.1) or more patterns (6-8 of 64,
near the memory's capacity) let spurious and other stored states fall below the cued one: then the lowest energy is
little better than one settle (e.g. 6 patterns, drive 0.1: 10.9% -> 22.7% at 16 tries, the oracle 84%). The energy
selects well exactly when the model's energy ranks its answers well - what the tries buy is bounded by that, not by N.


## Pinned allocation-free: a settle, a nudged pair, a lesson; repro/ trimmed (wt-pinned, 2026-11-21)

Every gemm workspace a system uses is reserved at Plan (settle.Engine.Ready), and ad/graph's Forward and Backward were
already pinned. Now pinned too, decided while compiling: a free settle and a nudged pair (dyn/system `settleOf`,
`nudgedOf`) and a whole EP lesson - the free settle, the nudged pair's certified rate and settle, the contrast
(dyn/learn `lessonOf`). The one thing in the way was `NudgedRate`'s two small arrays made per call (the coupling's four
block norms and the 2 x 2 matrix M), now kept on the System. ad/descent's settle is not pinned: it reads the state and
its gradient through `Graph.Value`/`Grad` each step, which makes views where a graph's views are stale and syncs from
the GPU - caching the arrays would skip that sync, so it stays as it is.

gemm with a storage type per operand (low-precision weights under F32 state) was already in kernels/gemm
(`Gemm<SC, SA, SB, A>`, BF16/F16/F8/INT8 weights widened as packed; the test "a storage type per operand ... bit for
bit", bench/storage).

repro/ re-checked against olang f73ca3c4f: every repro but `tempref` (still +72 kB over 100 000 calls) is fixed, and the
fixed ones are removed (git keeps them); `fieldname` stays as the record of a deliberate rule. Comments that named a
removed repro now say it was fixed.

## A two-start margin trainer and hybrid ranking (wt-rank, 2026-11-21)

`ad/descent`'s `Trainer.Margin(m)`: the state settled from two starts - where it is (the free fixed point) and from the
caller's target (`SetTarget`) - and, for every row whose two fixed points differ (by more than `Apart`) while its
energies are less than m apart, the hinge max(0, m + E_r(target's) - E_r(free)) stepped: dE/dtheta at the target's
fixed point less at the free one, averaged over the batch. Rows that do not step are given the same state in both
passes, so they cancel exactly; it needs the energy row by row (`RowEnergy`). Tested on a batch of tilted double
wells, each started in the wrong well: the gradient is the batch's mean hinge differentiated by central differences
(within 1e-6), and plain steps of it (rate 0.1) put the target's well m = 0.5 below the other in every row after 3
steps (the tilt 0.05 -> 0.45). It is the contrastive counterpart of EP for energies with several basins, where the
free settle's own fixed point is the wrong answer.

`nn/bestof`'s `BestOf.Rank(n, propose, score)`: one model proposes n candidates of every row, another scores them (an
energy, a verifier), the best kept per row - a fast proposer ranked by a slower energy. Tested on a proposer walking a
grid and a squared-distance scorer.

Its settled-pair form, `Descent.AddMargin(m, weight, start, right)` (wt-margin2): both states settled - the state as
it is and a second settle from `start` - the caller's `right(z, row)` judging each; a row with one right and one wrong
settled state whose energies are less than m apart adds weight (dE/dtheta(right) - dE/dtheta(wrong)) on top of what
ParamGrads holds, so it serves as a fine-tuning term beside Unrolled, Implicit or EP. Tested on the double wells: the
term is weight times the summed hinge's derivative (central differences, within 1e-6) added to the gradient already
there, and its steps put every right state m below the wrong one.

## Recorded higher-order autodiff: deferred

Decided 2026-11-21: a graph's backward stays its hand-written kernels; recording it as nodes (a derivative per
operation, plus a scalar-node broadcast the graph lacks) waits until an experiment needs exact second derivatives.
ad/descent's central differences (H v, (d g/dtheta)^T v) serve the trainers meanwhile, exact for quadratic energies.

## Faster settles on graph-structured states (wt-settlespeed, 2026-11-21)

`bench/graphsettle` (`make graphsettle`): a settle step on node states z [32, n, 16] - silu(z Wz^T + (A z) Wa^T + x Wx^T
+ b) read out to one number a node, plus a quadratic, A each graph's adjacency (3 random partners a node, both ways);
a step is a Forward and the Backward from the energy. One thread, us a step, medians of runs taken while the machine
was shared:

| n | dense A z, every gradient | message passing (SetDense) | and the state's gradient alone |
|---|---|---|---|
| 32 | 265-271 | 244-268 | 174 |
| 128 | 1563-2003 | 1099-1455 | 849-1078 |
| 256 | 4715-4874 | 2472-2660 | 1998-2037 |

1. `nn/message.Edges.SetDense(adj, graphs, n)`: the edges of a batch of dense adjacency matrices (weights kept,
   `Edges` now carries a weight per edge), so `Passing` with `Agg.Sum` is the batched product A z with work in the
   edges, not n^2: the dense product took 57% of a step at n = 256 (2.8 ms). SetDense scans the matrices once a batch
   (1.6 ms at n 256, batch 32: a settle of 16+ steps amortizes it; an edge list's `Set` is cheaper). Message passing
   itself is memory-bound at about 5 ns an edge on random graphs (lane operators measured level with the loop).
2. `Graph.Backward(loss, accumulate, params = false)`: only the watched state's gradient - the weights' products and
   column sums skipped, ParamGrads left as it was (tested bit for bit against a full Backward). `ad/descent` settles
   this way and asks for dE/dtheta once at the fixed point a trainer needs (EP's nudged phases); a settle step is 20-25%
   cheaper on its own.
3. kernels/gemm reads a transposed left operand in place when the result is narrow (`colsA`): a weight's gradient
   dy^T x with few outputs and a long depth no longer packs dy, which cost as much as its multiply-adds - 8192 x 32 x
   16 183 -> 140-152 us (46 -> 55-60 GFLOPS), Linear's backward at n 256 844-896 -> 761-776 us; wide products level.

For small transformers on the CPU (bench/lm, 4 x 128, ctx 64) nothing moved: the products run at 110-150 GFLOPS (near
a core's peak), attention at context 64 is 15% of a step at 9 GMAC/s (its tiny batched products and softmax - the next
target), and 4 threads were the lever (wt-gemmjoin).

## Checkpoints stream (wt-ckstream, 2026-11-21)

A large model's save ran out of memory: io/checkpoint's Writer kept every tensor encoded in memory and Bytes made the
whole file again before writing it, and File.Load read the whole file. Now a Writer encodes each tensor as it is added
into a scratch file (TMPDIR) a buffer of 64 KB at a time, Save writes the header and copies the scratch data after it
the same way (then removes it), and File.Load reads only the header and serves each tensor from the open file a buffer
at a time; File.Parse over bytes in memory and Writer.Bytes stay. The API is unchanged but Bytes (now fallible: it reads
the scratch back) and Discard (a writer given up without a Save). A test pins it: 64 MB of F32 saved and read back
raise the peak address space by 0 pages now, against 73732 pages (288 MB) with the old writer and loader.


## Muon's Newton-Schulz on kernels/gemm (wt-optfast, 2026-11-21)

Muon orthogonalized each matrix's step with hand-written triple loops (one walking a column stride), about 2 G
multiply-adds a second: a 128 x 128 matrix took 16.7 ms and a 512 x 512 one 1.37 s - several times a whole training
step of the models it is used on. Its three products a step (G = x x^T, b G + c G G, a x + P x) are now kernels/gemm's,
with the workspace in NsWork: 128 x 128 0.60 ms (28x), 512 x 128 1.78 ms (38x), 128 x 512 1.55 ms (39x), 512 x 512 27 ms
(50x), 1024 x 256 13 ms (35x), one thread. Muon's tests hold; a step of Muon and an Ema update and swap are now pinned
allocation-free (`assert not ... .Allocates`), beside AdamW's and Sgd's.

## Softmax rows: the maximum in lanes (wt-softmax, 2026-11-21)

A softmax row's running maximum ("v if v > top else top", one chain) is left scalar by the vectorizer and was most of
a row's cost: 1.4-2.2 ns an element against 0.5 for the exponentials beside it. `elem.MaxIn` keeps sixteen lanes and
folds them (the same value, NaNs as before), and SoftmaxRun / SoftmaxRunTo use it: a row of 64 0.74 ns an element (was
1.41), of 256 0.59 (was 1.91), bit for bit. Inside attention (4 heads, D 128, 1024 rows, causal): the softmax at context
64 364 -> 262 us, at 256 1278 -> 527 us; bench/lm's attention forward 4.0-4.4 -> 3.7-4.2 ms (its causal rows at context
64 average 32 elements, where little is gained). The products of attention at small context (15-24 GMAC/s, tiny batched
matrices) are what is left there.

## Batched key-value-cached decoding (wt-batchdecode, 2026-11-21)

Sampling many continuations (best-of-N proposers, reranking) ran the whole context through the model for every new
token: a graph of Batch sequences, one Forward over the context a token. A decoding graph now takes a batch: planned
with decode set and Batch > 1, its batched nodes are stored position by position ([ctx, batch, ...]), so one position of
every sequence is a run of rows that every operation reads as a batch; attention keeps a key-value cache per sequence;
positions and rotary angles count from the rows in use. Such a graph runs one position a Forward (views are refused by
name). `generate.CachedBatch(g, m, prompts, p, count, temperature, r)` prefills the prompts a position a step and then
samples every row a token a step. Tested: each row greedily the tokens its prompt alone decodes to, and every logit the
whole sequences' within 1e-12. `bench/sample` (`make samplebench`), one thread, random weights:

| model | batch | prompt + new | rerun | cached batch |
|---|---|---|---|---|
| 2 x 128, vocabulary 512 | 64 | 16 + 48 | 1145 tokens/s | 32754 tokens/s (28.6x) |
| 4 x 128, vocabulary 512 | 16 | 16 + 48 | 733 | 18973 (25.9x) |


## The settle's sweep: weights packed once a settle (wt-fusedsweep, 2026-11-21)

Item 1 of the kernel list (a fused settle sweep that keeps W on-chip across sweeps). On the CPU every sweep's carry was
a Gemm that packed W into its panels again - at a few rows that packing is most of the product, and W is read twice
from L2/L3 for one pass of arithmetic. Now `gemm.Packed<A>` holds op(B) in Gemm's panels and `GemmPacked` runs the
product on them (bit for bit Gemm's packed path with the same tile; a test, every transpose, block edges, threads);
each dense connection of a sweep carried with both sides at least 64 units wide gets room for its packed weights at
Plan, and a settle of more than four rows and two or more sweeps packs them once at its start (`Engine.PackAhead`, on
by default; dropped at its end, so learning between settles is free to change W). A population's reciprocal
connection to itself is carried both ways in one product: its two packed forms are laid out alike, so W^T + W is
their panels' sum - half the multiply-adds and half the weights read a sweep. A settle stays allocation-free; a test
holds packed and unpacked settles (Jacobi and Gauss-Seidel) to the same fixed point within 1e-9.

`bench/sweep` (`make sweepbench`): a hidden population of H tanh units coupled to itself (dense, reciprocal), a
readout of 10, fixed sweeps, one thread, us a sweep (packed every sweep -> once a settle):

| H | B 8 | B 32 | B 128 | B 512 |
|---|---|---|---|---|
| 128 | 15.4 -> 5.3 | 31.8 -> 15.0 | 94.7 -> 53.5 | 382 -> 226 |
| 256 | 49.0 -> 14.4 | 98.3 -> 42.7 | 276 -> 148 | 1239 -> 684 |
| 512 | 250 -> 46.2 | 463 -> 139 | 1285 -> 673 | 4821 -> 1990 |
| 1024 | 1858 -> 191 | 2454 -> 503 | 5358 -> 2060 | 21954 -> 8032 |

1.7-9.7x; the self-coupled case gains most at small batches (H 1024, B 8: 9.7x), where the old sweep spent its time
packing. A layered system gains only from the packing: the same system without the merge was 1.1-3.7x. The 784-128-10
MNIST system is unchanged (its in-sweep weights, 128 x 10, are below the 64-unit floor, where Gemm's exchanged tiling
wins). At four threads a sweep of one unit (batch up to 128 in one phase) is not faster: a product split over threads
there costs more in wakes than it saves (tried, reverted); batches of several units run their units in parallel as
before (H 1024, B 512: 2.1 ms a sweep).

On the GPU the same idea is one launch for all of a settle's sweeps with a region's W in shared memory; it needs the
local machine's RTX 4060 and is left for a session there.

## The chunked delta rule's workspaces (wt-deltachunk, 2026-11-21)

Item 2 of the kernel list (a chunked gated delta-rule scan). kernels/scan already had it in the WY form, forward and
backward; its chunked backward is 1.6-2.7x the sequential one (bench/scan: 4x1024, 4 heads of 64, 46.8 against 83.6
ms; 1x4096, 8 heads of 128, 233 against 630 ms), its forward about even at heads of 64 (1.4x at 128). ad/graph still
runs the sequential forms: its Forward and Backward are pinned allocation-free and the chunked form's `Allocates` fact
is true. Each task's gemm.Workspace is now reserved in the scan.Work (it was made at every call), the products' result
operands are locals, and still the fact holds: every part asserted alone is allocation-free and the resident memory
does not grow over 20000 calls (repro/chunkfacts.olang, sent to olang). Once the fact clears, the graph's DeltaRule
takes the chunked form.

## Flash attention (wt-flash, 2026-11-21)

Item 3 of the kernel list. `kernels/attention.FlashAttention` and `FlashAttentionGrad`: for each sequence and head the
queries are taken 128 at a time and the keys and values 128 at a time, each block's scores made by gemm and folded into
a running row maximum, sum and output (the online softmax; FlashAttention's tiling), blocks past a causal row never
made; the backward recomputes each block's weights from the row's log-sum-exp the forward keeps (one number a row and
head) - nothing Ctx x Ctx is stored. `Flash<A>` holds a gemm.Workspace and scratch a task; both calls are
allocation-free. Tested against Attention and AttentionGrad (F64 within 1e-11, a context of 150 against blocks of
128, one and three threads) and F32 against the F64 reference. ad/graph runs it for attention over a context of
`FlashFrom` (512) or more on the CPU, not decoding (the node keeps the log-sum-exp in its accumulator space; a test
holds a graph's values and gradients to the weighted form's within 1e-10).

bench/flash (`make flashbench`), causal, F32, ms forward / backward (weights kept by the weighted form):

| ctx, seqs x heads x dh | 1 thread: weighted | flash | 4 threads: weighted | flash | weights kept |
|---|---|---|---|---|---|
| 256, 8 x 4 x 32 | 3.9 / 5.6 | 3.3 / 6.3 | 2.4 / 6.2 | 1.7 / 2.9 | 17 MB |
| 1024, 2 x 4 x 64 | 20.0 / 38.4 | 15.8 / 32.4 | 6.3 / 12.2 | 3.9 / 7.8 | 67 MB |
| 2048, 1 x 8 x 64 | 76.4 / 137.6 | 63.8 / 123.2 | 23.6 / 41.1 | 16.0 / 32.3 | 268 MB |
| 4096, 1 x 4 x 64 | 150.4 / 294.4 | 113.3 / 218.8 | 41.5 / 82.6 | 28.7 / 56.7 | 537 MB |

Blocks of 64 x 64 were slower than the weighted form (6.0 / 8.3 ms at 256, one thread, with exp and the row maximum
scalar); elem's FastExp and sixteen-lane maximum and 128-wide blocks brought it level or ahead. In a training step
(bench/lm: 4 layers of 128, 4 heads, context 1024, 2 sequences, 4 threads) attention 86.7 -> 41.6 ms, a step 163 ->
118 ms, and the 268 MB of kept weights (four layers) gone.

## The sparse settle step (wt-sparsestep, 2026-11-21)

A settle over graph-structured states written with a dense adjacency product (`g.MatMul(A, z)`, A [batch, n, n]) pays
n^2 per graph whatever the degree. The edge-list form is nn/message: `edges := message.Edges(nodes, most)`,
`pass := message.Passing(edges, message.Agg.Sum)`, then `g.Custom(pass, z)` in place of the product - the same A z
(signed or weighted edges included), its backward A^T dz; each batch `edges.Set(src, dst, count)` from the edge
lists (SetDense reads dense matrices, but costs n^2 itself: 1.5 ms a batch at n 256). Its inner loops are now plain
per-element loops instead of slice expressions: forward 255 -> 178 us, backward 245 -> 198 us at n 256 (batch 32, D
16, degree about 6). Split over tasks and a pull-form backward by out-edges were tried and gained nothing (the pass
is memory-bound at these widths). bench/graphsettle, one thread, us a step (energy forward and backward):

| n | dense A z | edges | edges, the state's gradient alone |
|---|---|---|---|
| 32 | 235 | 226 | 172 |
| 128 | 1417 | 949 | 744 |
| 256 | 4305 | 2214 | 1789 |

What is left of a sparse step is the per-node dense layers (Linear about half), which no adjacency form changes.

## Weights packed once a descent settle: measured, not kept (2026-11-21)

The settle sweep's gain (dyn: weights packed once, 1.7-9.7x) does not carry to ad/descent's graph-energy settles.
Tried: every Linear's weights packed once per settle (forward and input-gradient forms, gemm.GemmPacked), a settle
step being a Forward and a state-only Backward. bench/graphsettle, one thread, us a step: n 32 x batch 32, D 16, H 32
(8192 rows, 32 x 16 weights) 182 -> 175, n 128 821 -> 842, n 256 2001 -> 1997; D 128, H 512, 64 rows 372 -> 336,
162 rows 822 -> 794; D 256, H 1024, 16 rows 365 -> 399. Packing is a small share of a product once it has 16 or more
rows, and those settles have many rows (nodes x batch); dyn's large gain came from sweeps of 8 rows and from merging
W^T + W. The step's cost there is the thin per-node products (k 16, n 32: about 26 GMAC/s) and the elementwise
operations, not the weights' packing. Reverted.

## Subnormals flushed at start-up, a fast exact GELU (wt-ftzgelu, 2026-11-21)

kernels/fpmode: `FlushDenormals` sets x86's flush-to-zero and denormals-are-zero (MXCSR, through a ten-line C file);
`Graph.Plan`, `Graph.PlanBaked` and `System.Plan` run `fpmode.Startup()`, which flushes unless `fpmode.KeepSubnormals`
was set first. The mode is per thread and olang's task workers are made at the first spawns that need them, taking
the mode of the thread that makes them - so a plan, made before a program's tasks, covers them (checked: four tasks
spawned after it all flush). A loop decaying toward zero through the subnormal range: 1.62 -> 0.06 ns an element
(27x). The suite is unchanged (413 passed).

The exact GELU (`mat.Act.Gelu`, x Phi(x)) read the C library's erfc, a call an element that blocks vectorizing: 8.5 ns
an element in F32 against 0.66 for the tanh form. `elem.FastErfc` is Numerical Recipes' erfcc (t exp(-z^2 + a degree-9
polynomial in t), relative error below 1.2e-7) on FastExp in F32 - within 1e-6 of the library's erfc in F64 for |z| <=
2, the error growing as z^2 beyond (exp(-z^2) of a rounded z, where GELU reads x times a number below 1e-3) - and the
library's erfc still in F64. GELU and its slope use it: 8.5 -> 1.5 ns an element.

## Ternary weights (wt-ternary, 2026-11-21)

Item 4 of the kernel list, for the FPGA stage. kernels/ternary: weights -1, 0 or +1 times a scale per output row
(BitNet b1.58's absmean quantization: scale the row's mean |w|, weights round(w / scale) clipped), stored as two
bit-planes of U64 words - 2 bits a weight, a sixteenth of F32 - and `MatVec` / `MatMul` that add, subtract or skip each
column's activation. The sum reads a byte of each plane at a time through a 256 x 8 table of 0/1 lanes (plus's less
minus's, times eight of x into eight lanes); reading bit by bit did not vectorize (0.25x of F32), and two sets of
lanes ran slower than one. Tested against the dequantized product. bench/ternary, one thread, us a product:

| rows x cols | F32 Gemv (weights MB) | ternary (MB) | ternary / F32 speed |
|---|---|---|---|
| 256 x 256 | 4.1 (0.3) | 8.8 (0.02) | 0.47 |
| 1024 x 1024 | 155.8 (4.2) | 121.5 (0.26) | 1.28 |
| 2048 x 2048 | 662.3 (16.8) | 462.1 (1.05) | 1.43 |
| 4096 x 4096 | 2876 (67.1) | 1841 (4.19) | 1.56 |

On the CPU it wins once F32's weights leave the caches (from about 1024 x 1024) and holds 16x the weights in the
same memory; in cache, F32's FMAs are faster. The form is the FPGA's: adder trees over selected activations.
nn/ternary trains such layers: `Quantizer`, an operation reading a latent F32 weight matrix as its absmean ternary values times each row's scale, its gradient passed straight through to the latent weights, and `Linear` over it, `Export` giving kernels/ternary's form with the same product (a test learns a ternary map to under 5% of its first loss and checks the exported MatVec against the graph's forward to 1e-5). 8-bit activations: below.

## Ternary connections in dyn (wt-dynternary, 2026-11-21)

`System.Ternary(c)` before Plan makes a dense or symmetric connection's weights ternary while the system learns: the
parameters hold each post row's absmean form (scale times -1, 0, +1) - what every sweep, energy, certificate and
adjoint reads, unchanged - and the connection keeps the latent real weights. A lesson's gradient (EP's contrast, the
implicit gradient) is taken at the ternary weights and stepped into the latent ones (straight through):
`System.Latent()` before an optimizer's step, `System.Quantized()` after it - learn.AdamW does both. Certify and
Restrain scale the latent weights with the parameters. CPU only (refused on the GPU at Plan).

First measure, XOR (2-32-2, the readout connection ternary, EP and AdamW at 0.03, 300 lessons): the loss falls (0.243
-> 0.130 for seed 2), every weight a row's scale times -1, 0 or +1 after every step (tested). Before the certificate's
scaling reached the latent weights, one seed of four solved XOR and three stopped at the same loss, 0.0867, with the
gradient near zero. Ternary weights learn slower than real ones here; how they fare on the experiments' tasks is
for the experiments. Wider is not better in this setting: with 128 hidden units (the same rate, 300 lessons) four
seeds ended at losses 0.50, 0.087, 0.130 and 0.50 (from 0.28-0.37), two of them rising - the rate tuned for 32 units
is too large for the latent weights of a 128-wide ternary row.

## A pairwise energy sampled by many chains (wt-potts, 2026-11-21)

nn/potts: a Potts energy over a sparse graph of K-valued cells - a K x K pair table shared by the edges and unary terms
linear in each cell's features - fitted by pseudolikelihood (each cell's conditional at a known answer exact, the
gradient for both tables in one pass, checked against central differences) and searched by `Sampler`: many chains,
each annealed by Gibbs sweeps over the graph's colour classes (a greedy colouring; tested that no class holds
neighbours) from T0 to T1, then zero-temperature sweeps until nothing moves, each chain stopping on its own; chains
on tasks, each with its own generator (the answers do not depend on the thread count, tested). A cell's pair energies
are kept as a field and updated only where a neighbour moves. `Langevin` is the relaxed counterpart (softmax logits
descended with annealed noise). bench/potts, the colouring energy on planted 3-colourable graphs (each cell joined
to 3 others), 32 chains of 300 annealing sweeps, ms a search:

| cells | 1 thread | 4 threads | best chain a proper colouring |
|---|---|---|---|
| 256 | 130 | 40 | 4 of 4 graphs |
| 1024 | 561 | 166 | 2 of 4 |
| 2048 | 1168 | 345 | 0 of 4 (more sweeps needed) |

About 50 ns a cell update at K 3, most of it the draw's exponentials (three of about 8 ns each); the field saved a
quarter. The sparse settle step at those sizes (bench/graphsettle, batch 8, D 16, one thread, a step): 256 cells 2078
-> 784 us, 512 4390 -> 1131, 1024 20866 -> 4051, 2048 110262 -> 5775 (19x) against the dense adjacency product.

## Integer spiking kernels (wt-spike, 2026-11-21)

kernels/spike: integer-only kernels for spiking and event-driven models on fixed-point hardware - fixed point with
saturating widths and a rounding shift, a sigma-delta encoder (a unit sends signed quanta of its change; what it has
sent tracks its value to within a quantum, tested), CSR fan-out and an event's scatter of integer weights times its
spike count (saturating at the accumulator's width), a table exp and a fixed-point softmax, a power-of-two leak, and
xorshift noise with an approximate normal. Moved here from an experiment, where they were written generic; on the
way the table exp's index was found rounding to nearest while each entry holds its interval's midpoint - a bias of
up to half a step (0.023 at d 0.14 with 32 entries a unit) - and now truncates (within 0.02 everywhere, tested).

## The graph's delta rule on the chunked form (wt-graphchunk, 2026-11-21)

olang 2e3a77a96 cleared the chunked delta rule's allocation fact (repro/chunkfacts, removed), so ad/graph takes it:
the backward always over a context of 64 or more, the forward where heads are 128 wide or more (at 64 the forms tie).
The chunked forms are pinned allocation-free. bench/deltagraph, a DeltaRule node, one thread, ms forward / backward:
4 x 1024, 4 heads of 64: 16.0 / 93.8 -> 15.4 / 55.5 (a step 1.55x); 1 x 4096, 8 heads of 128: 119.0 / 731.7 -> 91.6 /
274.6 (2.3x).

## nn/potts measured on Sudoku and colouring (wt-pottsspeed, 2026-11-21)

bench/potts now times a Sudoku case too (81 cells of 9 values, 20 peers each, 30 clues as unary features of a planted
grid, the rule's energy) beside planted 3-colourings, and the pseudolikelihood fit. Changes: the draw's exponentials
in one F32 pass, then their sum (Sudoku 175 -> 147 ns a cell update; colouring about 54 -> 49-54); the fit's gradient
skips zero features (Sudoku 394 -> 338 ns a cell). 32 chains of 300 annealing sweeps, 8 rounds a case:

| case | 1 thread: ms a search, ns an update | 4 threads | best chain at energy 0 | fit ns a cell |
|---|---|---|---|---|
| Sudoku 81 x 9 | 93.7, 147 | 26.2, 41 | 8/8 | 338 |
| colouring 256 x 3 | 121, 49 | 35.4, 14.3 | 8/8 | 70 |
| colouring 1024 x 3 | 535, 54 | 157, 15.8 | 5/8 | 88 |
| colouring 2048 x 3 | 1049, 53 | 332, 16.7 | 0/8 | 79 |

Where an update goes (colouring, 1024 cells): about 22 ns at zero temperature (no draw), 74 at T 1 (a draw and most
cells moving, each move updating its neighbours' fields). What would buy more: chains in vector lanes - a cell's
fields laid out across chains and the class swept for all chains at once - done next, below.

Chains in lanes: sixteen chains swept in step, a cell's fields, energies, draws (F32 FastExp) and moves computed for all
sixteen at once, each chain its own generator; a moving cell's change of W's row gathered once per lane and added to
every neighbour's fields as one vector multiply-add. The per-chain path is gone. Same bench, ns a cell update, one
thread: Sudoku 147 -> 40.7 (3.6x), colouring 256 49 -> 11.2, 1024 54 -> 11.4, 2048 53 -> 11.5 (4.7x); 4 threads (32
chains: two groups) Sudoku 25.5, colouring 9.0. Solved shares as before within the noise of 8 rounds (colouring 1024:
3/8 against 5/8, the chains' random streams now different).

## Ternary weights times 8-bit activations (wt-ternary8, 2026-11-21)

`ternary.MatVecI8`: the vector quantized by absmax to integers in -127 .. 127 (BitNet b1.58's activations), each row's
sum of those integers - adds and subtracts only, an FPGA's adder trees - times the row's scale and the vector's. Tested:
the integers are integers, the result is their exact sum scaled, and within the quantization's error of the F32
product. Two integer forms (I32 lanes times the bit tables; then mask tables, AND and add) ran 3.5-4x slower than the
F32 lanes on this CPU (their loops did not vectorize as well); held as F32, the integers' sums are exact up to 2^24 (132k
columns), so it runs MatVec's lanes: bench/ternary, one thread, us - 1024^2 126.5 (F32 activations 131.5, F32 Gemv
133), 2048^2 499 (497, 556), 4096^2 1998 (2003, 2314).

## Sparse settles from adjacency lists, and Langevin steps (wt-sparsesettle, 2026-11-21)

`message.Edges.SetAdjacency(adj, graphs, n)`: a batch's edges straight from per-node adjacency lists - no n x n matrix
read (SetDense's cost): 8 graphs of 256 nodes 796 -> 38 us, of 2048 nodes 59.4 -> 0.29 ms (205x); tested equal to
SetDense's edges. With it a settle over sparse graphs is `g.Custom(message.Passing(edges, Agg.Sum), z)` in place of
`g.MatMul(A, z)` and `edges.SetAdjacency(...)` each batch, the whole step then costing its edges (bench/graphsettle:
2048 nodes x 8, 110 -> 5.8 ms a step).

ad/descent: `Temperature` and `Cooling` - Langevin steps, Gaussian noise of variance 2 rate T after every settle step,
T multiplied by Cooling each step (annealing). Tested on E = |x|^2 / 2: each element's variance at a fixed T within
20% of the discretized chain's T / (1 - rate / 2), and cooling at 0.99 returns the state to the minimum.

`potts.Fitter`: pseudolikelihood fitting by Adam - Add each example of a batch (graph, features, known answer), Step
(the batch's mean gradient, W kept symmetric). Tested: from planted 3-colourings (4 graphs of 60 cells a step, 150
steps) the loss falls below 0.7 of its start and the fitted energy's chains find a proper colouring of a new graph of
120 cells.

`Sampler.Langevin` measured beside annealed Gibbs (bench/potts, 32 chains, 300 steps, T 0.3 -> 0.001, rate 1, one
thread): Sudoku 715 ns a cell step, the best chain's energy 31.6 on average (Gibbs: 43 ns, energy 0 on 6-8 of 8);
colouring 256 cells 73 ns, energy 19.3 (Gibbs 0). The relaxed search is the weaker and slower one on these discrete
problems; it is kept for continuous relaxations, not made faster.

`potts.ClueFeatures`: a constraint problem's cell features (a clue one-hot with a no-clue slot, the degree / 4) - with
them a Potts' B holds per-clue unary tables and a per-value degree term, the model e.g. a planted-colouring solver
fits; tested against the local energies written out.

## k-ary factors and hidden units in nn/potts (wt-pottsfactors, 2026-11-21)

`Factors`: factors over Arity cells, each a table of K^Arity energies (several tables, one per factor kind); `Hidden`:
binary hidden units joined to Arity cells each, an input C + sum of per-slot tables, summed out in the energy
(-softplus: Energy is the free energy at T 1) and in each cell's pseudo-conditional. Pseudolikelihood fits their tables
too (Fitter), the sampler draws every unit for all sixteen lanes before each sweep of the cells (block Gibbs: given
the units the cells couple only through pairs and factors), and `ColourFor` keeps a factor's cells in different
colour classes. Tested: the gradient of every table against central differences; the free energy against the sum over
the units' states; planted not-all-equal instances satisfied through 3-cell factors; hidden units fitted to
not-all-equal answers (the loss falls) and block-sampled. bench/potts, planted not-all-equal, 2 n clauses of 3, one
thread, 32 chains of 300 sweeps, ms a search / ns a cell update / satisfied:

| n | 3-cell factors | with two hidden units a clause beside them |
|---|---|---|
| 256 | 36.6 / 14.8 / 4 of 4 | 79.4 / 32.4 / 4 of 4 |
| 1024 | 166 / 16.7 / 2 of 4 | 331 / 33.5 / 0 of 4 |

(The units there carry small random tables: their cost, not a rule. A hand-set unit cannot penalize a clause: a unit
only lowers the free energy, so a not-all-equal rule through units has to reward the satisfied patterns - what a
fit finds.)

## A settle's units cut to the threads (wt-sweepcut, 2026-11-21)

A sweep's units are a population's rows in chunks of 128, so at batch 128 or less each population was one unit and a
system's other threads sat idle (bench/sweep, H 1024, B 128: 2.1 ms at 1 thread and at 4). Now a sweep that is split
over tasks (TaskWork, raised 2M -> 6M multiply-adds: below it the wakes cost more than they save) and has fewer units
than threads cuts them to about a thread's share of a batch (at least 16 rows); otherwise units stay 128 rows, so a
sweep run on one thread keeps its larger products. A row's sums do not depend on the cut (tested: results the same for
every thread count). bench/sweep, 4 threads, us a sweep (two runs each, the machine noisy to about 25%): H 1024, B 128
2056-2554 (1 thread) -> 845-861 (4 threads, was about 2200); H 256, B 128 189 -> 148; B 8 and 32 even with one thread
(they were up to 1.5x slower).

## Deep supervision in ad/descent (wt-deepsup, 2026-11-21)

An experiment had forked ad/descent for two additions; both are here now, so the fork can go. `Descent.Ds`: Unrolled(h)
with deep supervision - each unrolled step's state adds Ds C(x) to the cost differentiated, the gradient of C(x_T) +
Ds (C(x_T-1) + ... + C(x_T-h)) (tested against central differences of that sum over a 12-step settle). The fork's
Gaussian jitter of sd s after each step is `Temperature` = s^2 / (2 Rate) (Langevin steps, above).

A low-bit settle state, from a second experiment's fork: `StateBits` (above 0: after every settle step the state is
rounded per block of `StateBlock` elements - 0: a batch row - with a scale of its own: 1 bit sign times the block's mean
|x|; 2 bits -s, 0, s past 0.7 mean |x|; else symmetric levels, the scale min(max |x|, `StateClip` rms) / (2^(bits-1)
- 1); `StateStochastic` rounds stochastically from the noise generator), `ReadRounded` (the state kept exact, the
energy reading it rounded). Straight through: the unrolled gradient ignores the rounding. Tested: after a settle every
block lies on its levels (4 bits: whole multiples of its top / 7; 1 bit: plus or minus one value), and read-rounded
leaves the state off them.

## The settle as a sequence layer, fused (wt-seqsettle, 2026-11-22)

nn/seqsettle.SettleSeq: per token, K sweeps of s = tanh(W s + b_t) (W block-diagonal in heads), b_t = u_t + gam_t
Rot(phi_t) s_{t-1} pair by pair; cold or warm, carry or not; a graph operation with its exact backward through every
sweep (inputs u, gam, phi, W). The batch's sequences are vector lanes: states kept [sweep][neuron][token][sequence], a
sweep one broadcast multiply-add per weight over every sequence; tanh by FastTanh; the carry's sines and cosines a
polynomial after reduction to a quarter turn, the quadrant chosen by float arithmetic (with integer selects the loop
stayed scalar: 20 ns a pair); W's gradient one product a head and sweep at the end over every token and sequence (as
lane-wise sums inside the loop it ran at 3.4 GMAC/s). Tested against the layer written out token by token in F64 (cold
and warm, with and without the carry, one and three sweeps, angles to +-40): outputs within 2e-5, every gradient within
1e-4 relative. bench/seqsettle, batch 32, context 64, heads of N / 8, one thread, us a token forward / backward:

| N | K 1 | K 4 |
|---|---|---|
| 64 | 0.40 / 0.70 | 0.91 / 1.58 |
| 256 | 2.61 / 3.77 | 7.29 / 10.47 |

Against the same layer as scalar loops (an experiment's operation, run in the same harness): forward 6-7x faster
(N 256, K 1: 17.5 -> 2.8 us; K 4: 45 -> 7.2), backward about even (it was already lean; what is left is the strided
transposes in and out and the carry's backward).

Then (wt-seqsettle2): both sweeps' inner products in tiles of sixteen lanes held in registers over a row's weights (the
rows of st and gs had been loaded and stored for every weight); the carry's gains kept in the lane layout from the
forward and its gradients written there, transposed out once after the backward (they were strided gathers and
scatters every token, with a branch for token 0). Best of 3, us a token forward / backward: N 64, K 1 0.32 / 0.43;
K 4 0.59 / 1.11; N 256, K 1 2.06 / 2.74; K 4 4.67 / 6.50. Against the scalar loops in one harness: forward 7-12x,
backward 1.1-1.8x (the scalar backward was already about a quarter of its forward; ours is 1.3-1.9x our forward).

## Local learning rules for nn/potts (wt-pottslocal, 2026-11-22)

`Potts.Stats`: the energy's derivative for every table at an assignment (pairs, unary terms, factors, hidden units
summed out) - each entry sees only the cells it joins - tested against central differences of Energy. On it:
`Fitter.AddContrast` (the mean statistics of one set of chains less another's, times a scale), `Fitter.CD` (k Gibbs
sweeps at T 1 from the answer, from noise, or persistent chains a graph's own) and `Fitter.EP` (k free sweeps from
noise, then kn sweeps nudged toward the answer by +beta and by -beta through the sampler's new `Bias`, their contrast
over 2 beta); Step as for pseudolikelihood. Tested: CD-1, persistent CD-1 and EP each learn the colouring rule from
planted answers (40 % clues) and solve new graphs of 128 cells. bench/pottsfit, an experiment's colouring setting
(degree about 4, 20 % clues, 300 updates of 32 graphs of 32 cells, Adam 0.05 on a cosine, 16 chains a fit; tests of
20 chains x 400 sweeps, 16 graphs a size, seed 1): PL 0.07 s, CD-1 0.25 s, PCD-1 (a pool of 256) 0.19 s, EP (5 + 1
sweeps, beta 1) 0.82 s of training, and every rule 100 % of graphs at 128 and at 256 cells (the experiment's: PL
98-100, CD-1 97-99, PCD-1 85-87, EP 99-100 at 256).

## Propose and rank: discrete diffusion and a score-matched energy (wt-ebmrank, 2026-11-22)

Two generic pieces for the propose-and-rank pipeline. `nn/diffusion` - masked or uniform discrete diffusion over cells
of K values: `Corrupt` (a denoiser's training example at level t) and `Sample` (the reverse process in `Levels` steps,
with the exact posterior q(x_s | x_t, x0) for uniform noise and unmasking at rate (t - s)/t for masked), driven by any
denoiser callback; tested: corruption touches a share t of the cells, a perfect denoiser returns the answer, and
diffusion proposals ranked by a Potts energy through `bestof.Rank` keep the lowest-energy one. `ad/score` -
`ScoreMatch`, denoising score matching for an energy written on the graph: the parameter gradient of
sum |sigma dE/dz - eps|^2 as a central difference of dE/dtheta along u = 2 sigma (sigma dE/dz - eps) (two extra passes,
no second-order backward), tested against central differences of the loss.

## Hidden chains summed out by forward-backward (wt-hchain, 2026-11-22)

`nn/chain`: visible cells with one hidden state each, linked along a chain, energy B[z_0] + sum U[c_i][z_i][s_i] +
sum V[c_i][z_i][z_{i+1}] with tables indexed by each position's context. The free energy, all n x K conditionals (one
forward, one backward pass), and the pseudolikelihood gradient exactly - reverse mode through forward-backward,
O(n H (H + K)) where recomputing per cell and value costs O(n^2 K H^2) - and annealed ordered Gibbs scans with the
backward messages cached. Tested against brute force over the hidden chain (free energy, every conditional, the loss)
and central differences (the gradient). Addition (contexts a_i b_i, H = 2, batch 64 x 4000 steps, lowest PL loss of 3
starts, 3.9 s for the test): 32 / 32 12-digit sums exact after training on 1..4 digits, as the experiment that
introduced it (100 % at 12). Single starts can stall at PL 0.04 a digit (the latent half found), hence the restarts.

## A cheaper focused heat bath (wt-focusfast, 2026-11-22)

`Sampler.Focus` priced each move through the factor tables, walking every slot of every factor of a cell for each of
its values. Each chain now keeps every factor's table index with each slot open (`focusBase`), shifted on each move
(`focusMoved`): a cell's local energies read one entry a factor and value, and a factor's violation one entry. The
arithmetic is unchanged, so the results are bit-identical under a seed (`make focusbench`: every case's moves, solved
chains and answer checksum equal before and after). One thread, 4 chains: planted 3-SAT, 1000 and 4000 cells at clause
ratios 4.0 and 4.2, unweighted, 390-454 -> 260-295 ns a move (1.5-1.7x); with breakout weights 600-845 -> 355-433
(those runs are short, so setup counts); 3-colouring unchanged at ~110 ns (edges were already one read). Tried and
dropped: each cell's local energies cached until a neighbour moves (invalidation cost more than reuse: +5-15 %);
precomputed open indices per membership (no gain). Lanes do not pay at K = 2 (one value to price per cell). What is
left needs exact arithmetic dropped: per-cell energies updated by differences on each move (WalkSAT's break counts)
reorder the sums, so results would match only up to rounding.

Then (the Orchestrator agreed to results equal only up to rounding): each chain keeps every cell's local energies by
differences (`focusState`: a move updates its neighbours' and factor partners' energies and its constraints' place in
the violated set in one pass; a weight bump its cells'; a decay blends the energies with a copy kept at weight 1,
(1 - d) L1 + d L), recomputed exactly every 64 n moves (at least 65536). Test: after 2000 random moves, bumps and
decays the kept energies match a full recompute within 1e-9. ns a move, unweighted: 3-SAT 1000 x 4.0 / 4.2 207 / 208,
4000 x 4.0 / 4.2 276 / 249 (from 450 / 391 / 444 / 454 through the tables: 1.6-2.2x); colouring ~100 (from ~110-127).
Unweighted answers are unchanged under the seed; the weighted colouring runs differ by rounding. A recompute every
max(n, 1024) moves cost half of each move, hence 64 n. What remains (~200 ns, about 3 x 12 factor memberships a move)
is mostly scattered reads of the partner cells' energies and the factors' indices.

## Where Sampler.Anneal spends a factor-path sweep (wt-annealprof, 2026-11-22)

bench/potts gains a maze-shaped case: a 33 x 33 grid of 11-valued cells, horizontal and vertical pairs as two shared
2-cell factor tables (random, a glass). 37 ns a chain-cell update (32 chains x 2000 sweeps, one thread). Timed by
switching parts off: factorLanes (the table gathers) ~2 ns, the exponentials ~3 ns, the softmax draw with its pick
~23 ns together with the move bookkeeping. Fixed: a move computed each lane's change of W's row (k x 16 gathers) even
for a cell with no pairs - skipped now (37 -> 33.5 ns, same answers). Tried and dropped: an F32 copy of the factor
tables with I32 indices (44-47 ns); factor energies folded into the incremental field and updated lane by lane on
moves (52 ns: 35 % of lane visits move on this glass, so the scalar updates outweigh the gathers); the pick as a
branch-free count of running totals (50 ns) or as an F32 select (105 ns - the compiler's code for that loop, not the
work).

## A table for every pair, and exact search on forests (wt-pairs, 2026-11-22)

nn/potts `Pairs`: E(x) = sum_p W_p[x_I][x_J] + sum_i B_i[x_i], a K x K table per listed pair (`Pairs.All`, or
`Pairs.Screen`: the pairs whose cells' values share the most mutual information over solved examples). `PairFitter`
fits it by pseudolikelihood (Adam, L1 soft thresholds on tables and biases); `Pruned(thr)` keeps the pairs whose
table, with its row and column means taken out (what biases absorb), has an entry above thr. `ForestMin(clamp, x)`:
the exact lowest-energy free cells given the clamped ones by min-sum over the free cells' pair graph, refused
(`NotForest.Cycle`) unless edges = cells - components. Tests: 8 planted pairs among 45 over 10 three-valued cells,
3000 Gibbs samples - after 600 steps the planted couplings measure 1.50-1.84 and every other pair 0.05-0.30, so
`Pruned(0.5)` returns exactly the planted 8, and `Screen(8)` ranks them first; `ForestMin` equals brute force over
3^7 assignments on a tree tied to two clamped cells, and refuses once a cycle is freed.

## The energy pipeline as a public example (wt-pottslatin, 2026-11-22)

`examples/potts_latin.olang` (`make potts_latin`): Latin squares from solved examples, end to end on nn/potts; nn's
README gains a section on how the pieces work. Added for it: `Pairs.Model()` (the pairs as 2-cell factors for
Sampler) and `Pairs.Bias(clamp, hold)`; `FocusSchedule.Slack` (a constraint violated only when more than Slack above
its table's lowest - learnt tables are soft, so any excess counted every pair as violated and Focus never stopped:
0/50 before). Found on the way: a given held by too weak a bias (10 against pair costs of ~3.5 a conflict) yields to
three conflicting peers, and Focus never moves a cell outside a violated constraint, so its chains start at the
givens. One thread, 50 held-out puzzles, 40 % givens:

| n | pairs screened / kept / the rule's | fit | Anneal (16 chains x 400 sweeps) | Focus (16 chains, <= 20000 moves, Slack 1) |
|---|---|---|---|---|
| 6 | 270 / 197 / 180 | 0.35 s | 50/50, 6.4 ms | 47/50, 5.5 ms |
| 8 | 672 / 584 / 448 | 1.1 s | 42/50, 21 ms | 29/50, 86 ms |

## An exact large-neighbourhood move (wt-cutset, 2026-11-22)

nn/potts `Pairs.CutsetMove(x, region, count, r)`: the region's cells re-solved with the rest held - held cells cut at
random until at most one cycle remains among the free cells, then one cycle cell conditioned on each value and the
forest solved by ForestMin (cycle-cutset conditioning). x itself is a candidate, so the energy never rises. Tests: on a
6-ring with two pendants (K = 3) every start reaches the brute-force minimum over 3^8; on a 5 x 5 grid, 200 moves over
random 9-cell regions and the whole grid never raise the energy.

## Factors by their cells' sum, and block pseudolikelihood (wt-blockpl, 2026-11-22)

nn/potts `Factors.BySum`: a table indexed by the sum of its cells' values (each slot's step 1, `Stride` kept by Index;
`Width` the entries used), so Anneal, Focus and the fitters take it unchanged (bench/focus: every case's moves, solved
and checksums equal). `Fitter.AddBlocks`: block pseudolikelihood - each factor's scope (K^Arity joint assignments,
energies by changing its cells one at a time through Local) conditioned on the rest, cells in no factor alone; the
gradient is the scope's statistics at the data less their expectation (`scopeStats`, each term once). Test: 28
overlapping 3-cell sums over 30 cells (K = 3, 2.0 off the sum 3), 640 samples at T 1 - 300 steps of 8 examples
recover the gaps at 1.7-2.4, and 16 chains x 3000 sweeps on the fitted model satisfy every sum.

## Rules over whole lines: a hidden chain per line (wt-lines, 2026-11-23)

`nn/lines`: `Lines` - each line of cells carries a hidden chain of H states (shared tables A[h'][h][x], an end table
Z[clue][h] read at the line's clue, a bias B[x]), summed out exactly by forward-backward; the energy is the sum of the
lines' free energies. `PseudoGrad`: exact per-cell pseudolikelihood (a cell's conditional reads the lines through it;
the gradient is their expected counts at each of its values weighted by data less conditional); `Step` (Adam);
`Anneal` (single-site Gibbs on the summed free energy); `FitRestarts(restarts, fit, score, r)` keeps the restart a
caller's validation scores best. Tests: free energies equal brute force over the hidden paths, the gradient equals
central differences; run counts of the rows and columns of 4 x 4 binary grids (H 5, clues 0-2): 4 restarts x 1500
steps of 8 grids, the kept restart solves 8 / 8 validation puzzles and 32 / 32 held-out (best of 2 chains x 200
sweeps), 10.5 s.
