# oann - design

oann is a neural-network framework written in olang: tensors, kernels, reverse-mode autodiff, a dynamical-system core
for settling networks, models and their training loops, data, checkpoints. Every kernel is olang; nothing links a BLAS - but the GPU backend (`kernels/gpu`, cuBLAS, built only with
`-define Gpu=true`).
This file is the design as it stands (the cut-over, C3, 2026-11-15); the plan and every package's as-built notes are
`docs/redesign.md`, the model of settling networks is `docs/settling.md`, and the record of how the design got here -
rounds 1 and 2, before the redesign - is `docs/design-history.md`, the last section below says how to read it.

The yardstick is olang's own (PRINCIPLES.md in the olang repository): no manual memory management, C-like performance
with no cost the code does not show, natural language, minimal syntax. For oann: **a step, a sweep, a settle and a lesson
allocate nothing** (each has a test that runs steps and asserts the resident memory did not grow), every hot loop is a
product or one pass over memory, and a model is written the way a PyTorch or Flax user would expect.

oann is the public framework: everything general. The agent being bootstrapped on it - its configuration, weights,
created datasets and recipes - is oagent's (private), as is the first settling agent's machinery (`agent/` there).

## Layers

Each imports only the ones above it.

| layer | where | what |
|---|---|---|
| storage | `tensor.olang` | `Tensor<T>`: N-d strided views (rank up to 6), Slice, Select, Permute, Expand, Reshape, conversions between storage types (F64, F32, F16, BF16, F8, integers), `ShapeError` |
| kernels | `kernels/` | `Mat<S>` operands; `Gemm`, `GemmBatch`, `GemmLeft`, `Gemv`, `GemmI8` (generic over storage `S` and accumulator `A`), element kernels with stated errors, attention, scans (gated, selective, delta rule), sparse and event-driven products, convolution |
| autodiff | `ad/` | `Graph<T>`: record, `Plan` (one parameter region, one activation region reused by liveness), forward and backward; eager and captured runs agree bit for bit; a plan may be baked while compiling |
| dynamical systems | `dyn/` | `System<T>`: modules of populations (rate, spiking, custom neurons) with named state on `Columns`, connections (dense, symmetric, sparse, tiles, events, low rank), signals and rules on clocks; the settle, equilibrium propagation, three-factor and delta-write rules, energies, the bridge into `ad` |
| models | `nn/`, `models/` | layers, AdamW/Sgd and schedules over one flat region, the training loops, generation with a key-value cache, INT8 inference; the MLP, CNN, GPT, SSM and hybrid reference models |
| data | `data/`, `tokenizer.olang` | IDX, MNIST, a character corpus, token files, `Loader<D>` over a `Dataset` trait; byte-level BPE |
| IO | `io/` | safetensors by name (every dtype, read by numpy), JSON-lines run records |
| tools | `tools/`, `bench/` | per-operation gradient checks in F64, the drivers that measure each number of docs/redesign.md 4.3 |

`board/` is a backend of `dyn`, not a layer: a fixed-point engine (Q1.15 cells over saturating integers, the descriptor
table a packed struct) lowered from a `System`, the simulated FPGA; its golden vectors are digests.

## Rules the code keeps

1. **Shapes.** A shape a caller gives is checked once, where it is given (`Tensor.Over`, `Reshape`, `Plan`:
   `? ShapeError`); inside a planned graph or system a shape is oann's own invariant (`x!`). Kernels check nothing.
2. **Memory.** Everything is laid out by `Plan` and lives where the graph or system was made; eager mode allocates per op
   in the caller's block. Nothing assumes a planned structure is fixed: connections carry an active mask, sparse forms
   drop entries in place, populations retire units, so networks can be pruned while running.
3. **Precision.** Every kernel is generic over storage and accumulator: F32, BF16, F16, F8 or I8 storage, F32 (I32 for
   I8) accumulation, F64 for checks. A settle's state stays F32. Every reference model is tested in F32 and BF16.
4. **Threads.** Parallelism lives inside kernels and sweeps (`join`/`spawn`, the thread count fixed at `Plan`), never
   across graph nodes; results are the same for every thread count.
5. **Dispatch.** Built-in kinds are closed enums matched (ops, wirings, neurons, rules), `Custom(any ...&)` opens each
   set; per-element work is never behind a dispatch.
6. **Language features where they fit:** `Columns` for state, lane operators for element kernels, reflection for
   energies and checkpoint names, `comptime` for approximation coefficients and baked plans, typed attributes
   (key enums: `LifCell.Kept`, `Attn.Names`), `unique` with destructors for open files, packed structs for the board's tables.

## Tests and measures

`make test` runs every module's tests (the list is the makefile's `TESTS`). Each package's acceptance numbers, measured
against round 2's, are in docs/redesign.md 4.3 (M1 to M15) and its as-built notes; the programs that measure them are
`bench/` and `examples/` (`make` lists them with their arguments). `repro/` holds minimal programs for olang issues found
on the way.

## History

The design before the redesign is `docs/design-history.md` (the former DESIGN.md, sections 1 to 25): a recorded graph on
a 2-D `Matrix<T>` (round 1: 2.39 s an MNIST epoch to 0.69 by olang's own GEMM), transformers and BPE, mixed precision,
convolutions, INT8 inference, settling circuits with an agent, sleep and its nights. Where it names `linalg`, `nn`, `ops`,
`layers`, `optim`, `train`, `generate`, `conv`, `vision`, `quant`, `checkpoint`, `circuit`, `board`, `sparse`, `datasets`,
`agent`, `store` or `consolidator`, read the replacement in docs/redesign.md section 2; the files themselves are at tag
`pre-c3`.
