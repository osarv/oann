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
- Next here: the packed INT8 micro-kernel (the 64+-row gap) - it waits on `vpdpbusd` or a 16-bit packing that LLVM
  maps to one `vpdpwssd` per pair; ad/ops' `Scan` is `GatedScan` (the prelude now has a `Scan`).

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

**Compacting retired units out of the populations (a re-plan): not built**, larger than its worth now. What it needs:
a `System.Shrunk()` that records a new system from the old one's records (modules, populations at their live sizes,
signals, rules) and re-plans it; every connection rewired by the map of live units - dense, symmetric and event-driven
blocks as their live rows and columns, a low-rank one as A's live rows and B's live columns, sparse rows with their old
pattern restricted (Link has no way to take a pattern from outside `Draw` yet), tiles refused or turned dense (the tile
grid does not follow units); biases, the kept state fields (@Keep: thresholds, gains), drives, targets and labels
copied by the same map; the learner and optimizer made again. It saves the retired units' state and sweep work only,
which is small next to the weights `Sparsify` already removes.

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

## Next

- The open work of docs/redesign.md: the GPU route (question 1).
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
