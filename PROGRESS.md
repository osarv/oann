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

## Next

- The open work of docs/redesign.md: the GPU route (question 1) - GP1 to GP5 are done (above; G-M2's verdict: the
  settle on the GPU from batch x hidden 16384, `Check` 4); GP6 (`ad` on the GPU, above) is done. For GP7 G-M2 says the GPU's
  fixed cost a settle (about 1 ms whatever its size) sets the crossover: CUDA graphs or fewer readbacks would move it
  below 16384.
- INT8 on the GPU: a one-pass kernel dequantizing GemmI8's I32 product (kern.olang) instead of the second, F32-summed
  product (GP3: 192 against 2466 us at 2048^3); then G-M5.
- kernels/gemm with 4 threads is 2-4.5x slower than 1 thread at 32 and 128 rows of x 784 x 128 (G-M1, above).
- The nudged pair's certificate per batch: `NudgedRate` with the block norm certifies 60% of MNIST's rows at epoch 3
  (44% before; above), no batch of 32. Next lever: a budget per row (a per-row stop in dyn/settle), or a tighter
  hidden-to-readout block.
- Compacting retired units (above, if it pays).
- INT8: shallow products (the CNN's depth 9 and 144) and the per-lane `vpdpbusd`
  lowering (repro/vnnitile).

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
