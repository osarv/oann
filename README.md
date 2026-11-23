# oann

Neural networks in olang: N-d tensors, kernels, a recorded graph with automatic differentiation, layers, losses,
optimizers, transformers, state-space models, convolutions, INT8 inference, safetensors checkpoints, byte-pair tokens -
and settling networks, systems that learn by settling into a state (docs/settling.md), with a fixed-point engine for an
FPGA. Every kernel is olang; nothing links a BLAS. `DESIGN.md` is the design as it stands.

## Using oann

There is nothing to install or build: import oann's modules from your own program and olang fetches them once and
compiles them as part of your build (a remote import, `HOST/OWNER/REPO[@REF]/PATH`, the path a file without its extension;
`@REF` is a branch or tag, `olang.lock` pins the commit).

```
import "github.com/osarv/oann/tensor"

fn main() ? tensor.ShapeError {
    t := try tensor.Tensor<F32>.Of([1.0, 2.0, 3.0, 4.0], 2, 2)
    Println("shape " + t.Dims.Str() + ", t[1, 0] = " + $t.Get(1, 0))
}
```

Where a plan runs is chosen once, as PyTorch's `torch.set_default_device` does: on a build with the GPU
(`-define Gpu=true`, which the makefile passes wherever the CUDA toolkit is) and a machine with one, every plan made
after this line runs there, and on the CPU otherwise - a plan's `on` still decides for that plan (docs/redesign.md 1.10;
settling systems run on the GPU, `ad`'s graphs not yet):

```
import "github.com/osarv/oann/kernels/device"

device.SetDefault(device.Backend.Gpu if device.GpuAvailable() else device.Backend.Cpu)
```

Other modules the same way: `github.com/osarv/oann/nn/optim`, `.../ad/graph`, `.../models/gpt`, `.../dyn/system`.
Run a file by naming it - olang builds it fully optimized into its cache and runs it. The examples work as they are:

```
olang examples/xor_settle.olang                # XOR by a settling system, taught by equilibrium propagation
olang examples/mnist_mlp.olang 10 adamw 1 1    # the 784-128-10 perceptron on MNIST: epochs, optimizer, seed, threads
olang examples/charlm.olang 2000 1 2           # a character transformer on tiny Shakespeare
```

Each example's arguments are described at the top of its file (`examples/README.md`). MNIST and the text corpora are
fetched into `data/` on first use.

## Developing oann

The makefile is for oann's developers; `make` alone lists its targets, one line each with its ARGS, and runs nothing.
It needs the olang compiler beside oann (`../olang/build/olang`, or `OLANG=path`).

```
make test                          # every module's tests, takes minutes
make gputest                       # the GPU backend's tests (CUDA, cuBLAS; -define Gpu=true), skipped without a device
make gpusettle ARGS="5 10 20 4"    # the settle and a lesson on the GPU against the CPU at batch 32, 128, 512
make gpulearn ARGS="5 3"           # G-M2: mnist_learn's epochs CPU against GPU, batch 32-512, hidden 128 and 1024
make mnist ARGS="10 adamw 1 1"     # an example or benchmark built and run with ARGS (data, cnn, int8, lmbench, board, ...)
```

Downloads go to `data/`, everything built to `build/`; neither is in git.

## Layout

Each `.olang` file is one module, imported by its path relative to the importer (`import "../nn/optim"`).

| | |
|---|---|
| `tensor.olang` | `Tensor<T>`, the N-d strided view (rank up to 6): Slice, Select, Permute, Expand, Reshape, element kernels, conversions between storage types, `@` batched |
| `kernels/` | GEMM (blocked, batched, from patches, GEMV, INT8), element kernels, attention, scans, sparse and event-driven products, convolution; the backend a plan runs on (`device.olang`: `SetDefault`, `GpuAvailable`) and the GPU's in `kernels/gpu` (`kernels/README.md`) |
| `ad/` | reverse-mode autodiff on tensors: `Graph<T>`, the operations and their gradients (`ad/README.md`) |
| `dyn/` | the dynamical-system core: systems of populations and connections, the settle, equilibrium propagation and the other learning rules, energies, the bridge into `ad` (`dyn/README.md`) |
| `nn/` | layers, AdamW and Sgd, the training loops, generation, INT8 quantization (`nn/README.md`) |
| `models/` | the reference models: an MLP, a CNN, a GPT, an SSM, a transformer with a settle block (`models/README.md`) |
| `data/`, `tokenizer.olang` | IDX, MNIST, a character corpus, token files, the Loader over a Dataset trait; byte-level BPE |
| `io/` | safetensors checkpoints, JSON-lines run records |
| `bayes/` | Bayesian inference: Gaussian beliefs, Laplace / assumed-density updates for normal, logit, probit, binomial, Poisson and ordered likelihoods, drift, Kalman filter and smoother, quadrature, ADVI and HMC on `ad`, empirical Bayes, calibration (`bayes/README.md`) |
| `evo/` | evolutionary architecture search: schema and NEAT graph genomes, regularized evolution, tournament, (mu + lambda), NSGA-II, successive halving, weight inheritance, checkpoints, worker processes (`evo/README.md`) |
| `board/` | the FPGA engine simulated in integers, lowered from a `dyn` system (`board/README.md`) |
| `tools/` | gradient checks per operation (`tools/README.md`) |
| `examples/` | the programs the makefile runs: MNIST, transformers, settling systems, INT8, the board (`examples/README.md`) |
| `bench/` | benchmarks and the checks against reference implementations (`bench/ref/`) |
| `docs/settling.md` | settling networks: the model, the papers it rests on, the open questions |
| `repro/` | minimal programs for olang issues found here (`repro/README.md`) |

The settling agent (its memory store, sleep, the bandit and nights experiments) lives in oagent's `agent/`; what a library needs of it is `dyn`'s rules.
