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
after this line runs there, and on the CPU otherwise - a plan's `on` still decides for that plan (docs/redesign.md 1.10).
`ad`'s graphs train there in F32, F64 or BF16 (mixed precision: F32 masters and moments on the device); what has no GPU
form yet (dropout, convolutions, scans, a custom operation, decoding) is refused at `Plan` by name:

```
import "github.com/osarv/oann/kernels/device"

device.SetDefault(device.Backend.Gpu if device.GpuAvailable() else device.Backend.Cpu)
```

That GPU is NVIDIA's (CUDA, cuBLAS). Other cards - AMD, Intel, integrated GPUs, anything with a Vulkan driver - are
reached through the Vulkan backend, `kernels/vk` (docs/vulkan.md, `-define Vulkan=true`): the products and the kernels a
transformer's inference needs, called directly, not yet through `device.SetDefault`, and open-weight language models
decoding on it (`oann-gen --device vulkan`, llm/README.md). It is there but lightly tested: its tests
run on software Vulkan, an NVIDIA RTX 4060 and an AMD Radeon integrated GPU (Windows); Intel's and phones' have not
seen it - so expect bugs.

Other modules the same way: `github.com/osarv/oann/nn/optim`, `.../ad/graph`, `.../models/gpt`, `.../dyn/system`.
Run a file by naming it - olang builds it fully optimized into its cache and runs it. The examples work as they are:

```
olang examples/xor_settle.olang                # XOR by a settling system, taught by equilibrium propagation
olang examples/mnist_mlp.olang 10 adamw 1 1    # the 784-128-10 perceptron on MNIST: epochs, optimizer, seed, threads
olang examples/charlm.olang 2000 1 2           # a character transformer on tiny Shakespeare
```

Each example's arguments are described at the top of its file (`examples/README.md`). MNIST and the text corpora are
fetched into `data/` on first use.

## Generating text with open-weight models

`llm/` loads Hugging Face checkpoints of the Qwen2 and Qwen3 architectures (Qwen3-0.6B to 8B, DeepSeek-R1-Distill-Qwen-1.5B
and 7B, Qwen2.5) and runs them in batches on the CPU - BF16 weights on AMX tiles where the CPU has them, F32 elsewhere;
`--int8` stores the layers' weights in 8 bits. `oann-gen` writes completions of a JSON-lines file of prompts as they finish, and picks up
where it left off when run again:

```
make oanngen
build/oann-gen Qwen/Qwen3-1.7B prompts.jsonl out.jsonl --n 4 --temp 0.6 --max-tokens 4096
```

A model named by its repository is fetched once from Hugging Face; its licence is the model's (Apache 2.0 for Qwen3,
MIT for the DeepSeek-R1 distillations). How it works, the flags, the checks against the reference implementation and
the speed on a 4-core machine are in `llm/README.md`.

## Distributed training

`nn/dist` trains one model on machines that share no network but a store of blobs - cloud VMs with outbound HTTPS
only, or processes on one machine - by DiLoCo (Douillard et al. 2023): every worker trains a whole copy with its own
optimizer for H steps, hands in its delta (the shared model minus its own) and takes back the outer step - Nesterov
momentum on the average of the deltas - that the coordinator, worker 0, published. A round moves one delta up and one
update down a worker, each the size of the parameters in the codec chosen (BF16: half the F32 parameters), instead of a
gradient every step.

```
import "github.com/osarv/oann/nn/dist"

params := g.Params().Flat()!                        # a BF16 graph: g.Masters(), and g.SyncParams() after a Sync
d := dist.DiLoCo<dist.DirStore>(dist.DirStore("/shared/run1"), worker, workers, params.Len())
try d.Begin(params)                                 # every worker from the same seed; a late one from the newest state
for s in range 1, steps + 1 {
    train.LmStep(g, m, data, o, rate(s - 1), 1.0)   # the inner optimizer, whatever it is, keeps its own state
    if s % H == 0: try d.Sync(params)               # a Report: the round, its contributors, bytes up and down, the wait
}
```

`make diloco ARGS="diloco 4 2000 10"` runs four worker processes of `examples/diloco_lm.olang` through a directory;
`"single 2000"` and `"dp 4 2000"` are the baselines it is measured against. Options of `DiLoCo`: `outerRate` and
`momentum` (0.7, 0.9), the deltas' and the update's codec (`coded`, `updates`: F32, BF16, Int8, TopK with `keep`),
`fragments` (streaming: a round syncs one fragment, so a sync every H / fragments steps moves that share of the bytes),
`timeoutMs` and `quorum` (a worker not in by then is left out of the round and its delta folded into that fragment's
next round), `snapshotEvery` (a state every so many rounds, for joining and resuming) and `name`/`lineage` (into the
state's metadata).

Measured on one 4-core VM (`examples/diloco_lm.olang`: the 0.81M-parameter character transformer on tiny Shakespeare,
batches of 16 a worker, every worker one thread; DiLoCo's workers four processes through a directory; BF16 deltas
unless said; seconds of training, evaluation left out):

| run | tokens | seconds | validation loss | bytes a round (coordinator up / down) |
|---|---|---|---|---|
| one worker, 2000 steps | 2.0M | 158 | 1.844 | |
| one worker, 8000 steps | 8.2M | 649 | 1.500 | |
| data parallel, 4 replicas of 16, 2000 steps | 8.2M | 221 | 1.562 | |
| DiLoCo 4 workers, H = 250 | 8.2M | 164 | 1.806 | 3.2 MB / 4.9 MB |
| DiLoCo, H = 125 (F32 / Int8 deltas) | 8.2M | 173 (187 / 176) | 1.757 (1.755 / 1.759) | 3.2 / 4.9 (4.9 / 9.7, 2.4 / 2.5) |
| DiLoCo, H = 125 in 5 streamed fragments | 8.2M | 191 | 1.756 | 0.65 / 0.97 |
| DiLoCo, H = 125, TopK 5% | 8.2M | 170 | 1.875 | 1.9 / 0.7 |
| DiLoCo, H = 50 (outer rate 0.4 / 1.0) | 8.2M | 171 | 1.672 (1.752 / 1.648) | 3.2 / 4.9 |
| DiLoCo, H = 50, plain averaging (rate 1, no momentum) | 8.2M | 171 | 1.773 | 3.2 / 4.9 |
| DiLoCo, H = 25 | 8.2M | 190 | 1.610 | 3.2 / 4.9 |
| DiLoCo, H = 10 | 8.2M | 196 | 1.563 | 3.2 / 4.9 |

Four DiLoCo workers beat one worker at the same wall time at every H, and at H = 10 match data parallel at equal
tokens. This run is short (2000 steps, from scratch), where fewer steps between rounds pay most; the paper's H of
hundreds is for long runs from a warm start. The outer Nesterov step matters (1.65-1.67 against plain averaging's 1.77),
Int8 deltas cost nothing measurable, streaming keeps the loss at a fifth of the bytes a sync, and TopK at 5% does not
pay here.

The store is a trait of three methods - `Put(name, bytes)`, `Get(name)` (Missing for a name not there yet) and `Flush()`
(make what was put visible at once) - so a backend is a struct with those. Blobs are written once and never changed;
names are `RUN/r<R>.d<W>` (worker W's delta of round R), `RUN/r<R>.u` (the update; its contributors in the metadata),
`RUN/state.<R>` (the global model and outer momentum after round R) and `RUN/latest` (the newest state's round, as
text), RUN the DiLoCo's `name`. Each is a safetensors file. The backends:

- `DirStore(root)`: files in a directory (processes on one machine, or a shared disk).
- `HttpStore(base, token)`, any blob service: `PUT base/NAME` with the bytes as the body (any 2xx is success), `GET
  base/NAME` giving them back (404 while there is none), `Authorization: Bearer TOKEN` when a token is given; a PUT must
  become visible whole (a GET sees all of a blob or a 404, never part of one), as object stores' PUTs do. An S3-style
  bucket through token-authorized or presigned URLs, or a small relay in front of any object store, meets it as it is.
- `HfStore(repo, token)`, a Hugging Face repository (`kind` "model" or "dataset"), where an upload is a commit: `Flush`
  commits what was put in one commit - a worker's delta, or the coordinator's update with its state, so a round is one
  commit a worker - small blobs inline, large ones first to Git LFS (single or multipart uploads, as the Hub's preupload
  check and LFS batch API say); `Get` reads `resolve/REVISION/NAME`. The token needs write access to the repository.
  The Hub limits commits a repository an hour, so it suits rounds a minute or more apart; faster rounds want an object
  store. Tested against a local mock of those endpoints (inline, LFS, multipart, a late joiner, a refused token).

A state can be a surgery checkpoint (`surgery/model`'s format): given the model's `Net.Layout()` as `Named` and its
`State.DistMeta()` as `Meta`, the coordinator writes `p.NAME` for each parameter and `outer.NAME` for its outer
momentum, with the spec, step, data position and lineage (the round appended). `model.State.Open` reads it,
`surgery/ops` grows or prunes it - the outer momentum moved with its parameters as a first moment is, kept by
`ResetOptimizer` - and `DiLoCo.Publish(net.G.Params(), net.Outer(st), round)` seeds the grown model's run under a new
name, numbered on, for the workers to `Begin` from.

## Developing oann

The makefile is for oann's developers; `make` alone lists its targets, one line each with its ARGS, and runs nothing.
It needs the olang compiler beside oann (`../olang/build/olang`, or `OLANG=path`).

```
make test                          # every module's tests, takes minutes
make gputest                       # the GPU backend's tests (CUDA, cuBLAS; -define Gpu=true), skipped without a device
make vktest                        # the Vulkan backend's tests (any GPU, or software Vulkan), skipped without a device
make vkgemm ARGS="5"               # the product on the Vulkan device against the CPU
make vkllm ARGS="MODEL_DIR 64 1 8" # a language model's decoding on the Vulkan device against the CPU, then tokens a second
make whispervk ARGS="small 3"      # Whisper's encoder and a decoder step on the Vulkan device against the CPU
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
| `nn/` | layers, AdamW and Sgd, the training loops, DiLoCo over a blob store (`nn/dist`), generation, INT8 quantization (`nn/README.md`) |
| `models/` | the reference models: an MLP, a CNN, a GPT, an SSM, a transformer with a settle block (`models/README.md`) |
| `data/`, `tokenizer.olang` | IDX, MNIST, a character corpus, token files, the Loader over a Dataset trait; byte-level BPE |
| `io/` | safetensors checkpoints, JSON-lines run records |
| `bayes/` | Bayesian inference: Gaussian beliefs, Laplace / assumed-density updates for normal, logit, probit, binomial, Poisson and ordered likelihoods, drift, Kalman filter and smoother, quadrature, ADVI and HMC on `ad`, empirical Bayes, calibration (`bayes/README.md`) |
| `evo/` | evolutionary architecture search: schema and NEAT graph genomes, regularized evolution, tournament, (mu + lambda), NSGA-II, successive halving, weight inheritance, checkpoints, worker processes (`evo/README.md`) |
| `surgery/` | changing a trained model instead of retraining it: a checkpoint of spec, named parameters, optimizer state, positions and lineage; function-preserving growth (deepen, widen, heads, vocabulary, context), removal by importance, distilled replacement, optimizer conversion (`surgery/README.md`) |
| `archnet/` | a network for architectures, learned online: genome encoders, a performance predictor with uncertainty and curve extrapolation, a generator of genomes; the advisor evo's searches take (`archnet/README.md`) |
| `llm/` | inference for openly licensed language models (Qwen2 and Qwen3 checkpoints from Hugging Face): byte-level BPE from tokenizer.json, BF16 or INT8 weights on AMX tiles, a paged key-value cache, continuous batching, sampling, and `oann-gen`, completions of a file of prompts as one resumable job (`llm/README.md`) |
| `board/` | the FPGA engine simulated in integers, lowered from a `dyn` system (`board/README.md`) |
| `tools/` | gradient checks per operation (`tools/README.md`) |
| `examples/` | the programs the makefile runs: MNIST, transformers, settling systems, INT8, the board (`examples/README.md`) |
| `bench/` | benchmarks and the checks against reference implementations (`bench/ref/`) |
| `docs/settling.md` | settling networks: the model, the papers it rests on, the open questions |
| `repro/` | minimal programs for olang issues found here (`repro/README.md`) |

The settling agent (its memory store, sleep, the bandit and nights experiments) lives in oagent's `agent/`; what a library needs of it is `dyn`'s rules.
