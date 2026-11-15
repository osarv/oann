# oann

Neural networks in olang: N-d tensors, kernels, a recorded graph with automatic differentiation, layers, losses,
optimizers, transformers, state-space models, convolutions, INT8 inference, safetensors checkpoints, byte-pair tokens -
and settling networks, systems that learn by settling into a state (docs/settling.md), with a fixed-point engine for an
FPGA. Every kernel is olang; nothing links a BLAS. `DESIGN.md` is the design as it stands; `docs/redesign.md` is the
plan and what each package measured.

## Building and testing

oann needs the olang compiler: check olang out beside oann and run `make` there, which bootstraps `build/olang`.

```
make test                          # every module's tests (OLANG=path/to/olang names another compiler)
make mnist ARGS="10 adamw 1 1"     # the 784-128-10 perceptron on MNIST: epochs, optimizer, seed, threads
make charlm ARGS="2000 1 2"        # a character transformer on tiny Shakespeare
```

The makefile lists every program with its arguments (data, cnn, bpelm, lmbench, int8, board, mnistlearn, ...).
Downloads go to `data/`, everything built to `build/`; neither is in git.

## Layout

Each `.olang` file is one module, imported by its path relative to the importer (`import "../nn/optim"`).

| | |
|---|---|
| `tensor.olang` | `Tensor<T>`, the N-d strided view (rank up to 6): Slice, Select, Permute, Expand, Reshape, element kernels, conversions between storage types, `@` batched |
| `kernels/` | GEMM (blocked, batched, from patches, GEMV, INT8), element kernels, attention, scans, sparse and event-driven products, convolution (`kernels/README.md`) |
| `ad/` | reverse-mode autodiff on tensors: `Graph<T>`, the operations and their gradients (`ad/README.md`) |
| `dyn/` | the dynamical-system core: systems of populations and connections, the settle, equilibrium propagation and the other learning rules, energies, the bridge into `ad` (`dyn/README.md`) |
| `nn/` | layers, AdamW and Sgd, the training loops, generation, INT8 quantization (`nn/README.md`) |
| `models/` | the reference models: an MLP, a CNN, a GPT, an SSM, a transformer with a settle block (`models/README.md`) |
| `data/`, `tokenizer.olang` | IDX, MNIST, a character corpus, token files, the Loader over a Dataset trait; byte-level BPE |
| `io/` | safetensors checkpoints, JSON-lines run records |
| `board/` | the FPGA engine simulated in integers, lowered from a `dyn` system (`board/README.md`) |
| `tools/` | gradient checks per operation (`tools/README.md`) |
| `examples/` | the programs the makefile runs: MNIST, transformers, settling systems, INT8, the board (`examples/README.md`) |
| `bench/` | benchmarks and the checks against reference implementations (`bench/ref/`) |
| `docs/settling.md` | settling networks: the model, the papers it rests on, the open questions |
| `docs/redesign.md` | the redesign: architecture, module map, work packages and what each measured |
| `docs/design-history.md` | the design record of rounds 1 and 2, before the redesign |
| `repro/` | minimal programs for olang issues found here (`repro/README.md`) |

The first settling agent (its memory store, sleep, the bandit and nights experiments) was built on the circuit this
design replaced and lives in oagent's `agent/`; what a library needs of it is `dyn`'s rules.

`PROGRESS.md` is the hand-off note of the work in progress. `git show pre-c3:PATH` has any file the cut-over removed.
