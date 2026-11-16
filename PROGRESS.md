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
- Not built: pruning `Tiles` by tile and `LowRank` by rank, compacting retired units out of the populations (a re-plan);
  `Sparsify` keeps an optimizer's moments only if the caller remakes it.
- Found on the way: a skip connection from the inputs straight into the readout (x -> y, folded) stops this system
  from learning XOR at all (52%), with or without `Restrain`; not chased.

## Next

- The open work of docs/redesign.md: the GPU route (question 1).
- Pruning: a schedule that slows near the target (the 10% collapses above come at the last ticks), and a test of
  pruning on MNIST (`mnist_learn`).

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
