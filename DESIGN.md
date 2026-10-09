# oann - design

oann is a neural-network library written in olang: a recorded graph with automatic differentiation, layers, losses,
optimizers, datasets and the training loop, on top of the standard library's 2-D `Matrix<T>` (std/linalg) and its
seeded generator (std/rand). The C version it replaces is gone; `bench/ref/mlp.c` is the one C file left, a reference
trainer over OpenBLAS written for the benchmark.

The yardstick is olang's own (PRINCIPLES.md in the olang repository): no manual memory management, C-like performance
with no cost the code does not show, natural language, minimal syntax. Concretely for oann: **a training step
allocates nothing of its own**, every hot loop is a GEMM or one pass over memory, and a model is written the way a
PyTorch or Flax user would expect.

**State (phase 3):** the graph, its kernels, layers, softmax cross-entropy, SGD and both AdamW variants are built, and
the 784-128-10 perceptron trains on MNIST to 97.7-97.8% test accuracy in 10 epochs (phase 2). **Transformers are
built** (section 11): tokens, embeddings, learned positions, layer and RMS normalization, fused multi-head attention,
dropout, padding rows, GPT-2's blocks, a warmup-then-cosine schedule, gradient clipping, checkpoints and generation
with a key-value cache - every backward checked against central differences, the first training steps equal to the same
model in numpy to six decimals, and a character-level model trained on tiny Shakespeare. **Next: settling networks**
(docs/settling.md).

## 1. The operand: Matrix

Every value oann computes with is a `linalg.Matrix<T>`: **rows are samples (the batch), columns are features**. There
is no N-d tensor - data is interpreted in one place only, by the operation that consumes it.

- **A sample is a row.** A dense layer maps `[B, in]` to `[B, out]`; a loss reads one row per sample.
- **Weights are `[out, in]`** (PyTorch's layout, so checkpoints map one to one onto PyTorch's): a dense layer is
  `y = x W^T + b`, its backward `dx = dy W`, `dW = dy^T x`, `db` = the column sums of `dy` - each one `linalg.Gemm`
  with transposes as parameters. Nothing is ever transposed in memory.
- **A bias is a `1 x C` matrix**, added to every row by an explicit row operation; there is no implicit broadcasting.
- **Views are by row stride.** A matrix is shape, stride and a reference to storage; `linalg.View(data, offset, rows,
  cols)` makes one over the graph's arena without allocating, and a short batch is the view of its first rows.
- **Structure beyond two dimensions lives in the operation.** A convolution reads each row as one `C x H x W` image
  (im2col into a workspace, one GEMM); attention reads rows as `B x T` tokens and its heads as column blocks
  (section 11).

oann is generic over the element type: MNIST trains in `F32`, the gradient checks run in `F64`, and `BF16` storage with
`F32` accumulation is the mixed-precision path (std/linalg already accumulates `F16`/`BF16` products in `F32`).

## 2. Automatic differentiation: a recorded graph, replayed every step

The model is run **once**, against a `nn.Graph<T>`: every operation appends a node (an `Op` and its inputs' handles)
and gives its handle, a `Var`, back. `g.Plan(r)` then decides which nodes need gradients, lays every value, gradient
and saved intermediate out in **one arena**, allocates it once and draws the parameters' starting values from `r`. A
training step is `Forward(n)` (the nodes in recorded order), `Backward(loss)` (in reverse, each op's vector-Jacobian
product) and the optimizer's `Step(g)`, none of which allocates.

Each primitive has a hand-written backward (in `ops.olang`); the graph composes them, so a residual connection, shared
parameters or a second loss need no new backward code.

### The alternatives, and why not

1. **Layer-level backward** (the old C version, Caffe, Darknet): every layer writes forward and backward over its whole
   computation, and a container chains them. It does not compose - a residual, a shared embedding or two losses need
   special layers - and without run-time interfaces the container would be an enum of every layer type.
2. **An eager tape** (PyTorch's define-by-run): ops run as they are called and record themselves. olang's arenas
   would make that cheap (one scope per step), but every step would rebuild every activation, and nothing could be
   planned once: no buffer layout, no fusion, no thread decisions. Kept possible (section 12): the op set and kernels
   are shared, so an eager tape can be added beside the graph.
3. **A recorded static graph** (JAX's jit, XLA, TFLite's arena planner, tinygrad's schedule) - the choice. Shapes are
   fixed per graph, which is what MLPs, CNNs and transformers trained at a fixed batch and sequence length have; a new
   shape is a new plan.

### Why it fits olang

- **Handles, not references.** `type Var I64` is nominal, so a handle is never mixed up with a count. A `Node` is plain
  data (`Kind Op`, its shape, its value's, gradient's and saved matrix's offsets); the graph holds a `List<Node>` and
  one arena `Array<T>&`, nothing in it refers to anything else, so the scope checker has nothing to prove.
- **The op set is a closed enum dispatched by `match`** (a switch - no indirect calls):
  `MatMul(a, b, ta, tb)`, `Linear(x, w, b)`, `AddRow(x, b)`, `Add`, `Sub`, `Mul`, `Scale(x, k)`, `Relu`,
  `LeakyRelu(x, slope)`, `Gelu`, `GeluTanh`, `Sigmoid`, `Tanh`, `Softmax`, `Embedding(tokens, table)`,
  `Positions(x, table, T)`, `LayerNorm(x, gain, bias, eps)`, `RmsNorm(x, gain, eps)`, `Attention(q, k, v, heads, T,
  causal)`, `Dropout(x, p)`, `SoftmaxCrossEntropy(logits, classes)`, `Mse`, and the leaves `Input`, `Classes`,
  `Tokens`, `Constant`, `Param(init, decay)`.
- **Views are values.** `g.Value(v)` and `g.Grad(v)` are `Matrix<T>` views of the arena, made per use for nothing.
- **Traits only as constraints.** A training loop takes `o mut <O optim.Optimizer<F32>>&`, so the optimizer's step is a
  direct, inlinable call.

### Graph semantics

| oann | meaning | PyTorch analogue |
|---|---|---|
| `g := nn.Graph<F32>(128, threads)` | a graph planned for batches of up to 128 rows; products split over `threads` tasks | - |
| `x := g.Input(784)` | a `[B, 784]` input the caller fills each step (`g.InputData(x)`) | a batch tensor |
| `y := g.Classes()` | `B` class indices, `I32` (`g.ClassData(y)`) | `targets` |
| `t := g.Constant(r, c)` | a matrix the caller sets and nothing trains: a target, a mask | a tensor without grad |
| `w := g.Param(out, in, init, decay)` | a trainable matrix; `decay`: weight decay applies to it | `nn.Parameter` |
| `g.Linear(x, w, b)`, `g.Product(a, b, ta, tb)`, `g.Multiply(a, b)`, `g.Relu(h)`, ... | record an op, give its handle | ops in `forward` |
| `g.Plan(r)` | lay out the arena, allocate it, initialize the parameters - once | `torch.compile` |
| `g.Forward(n)` | run the batch's first `n` rows | `model(x)` |
| `g.Backward(loss)` | parameter gradients of `loss`, written fresh | `zero_grad(); loss.backward()` |
| `g.Backward(loss, true)` | added to the gradients already there | `loss.backward()` without zeroing |
| `g.Value(v)`, `g.Grad(v)`, `g.Scalar(loss)` | views of a node's value and gradient; a loss's number | `.data`, `.grad`, `.item()` |
| `g.Params()`, `g.ParamGrads()` | every parameter (or gradient) as one flat matrix | `parameters()` flattened |
| `t := g.Tokens()` | a token per row, `I32` (`g.TokenData(t)`) | `input_ids` |
| `g.Embedding(t, table)`, `g.Positions(x, table)` | gather the tokens' rows; add row `(Pos + r) mod T` | `nn.Embedding` |
| `g.LayerNorm(x, gain, bias)`, `g.RmsNorm(x, gain)` | row-wise normalization | `nn.LayerNorm`, `nn.RMSNorm` |
| `g.Attention(q, k, v, heads, T, causal)` | multi-head attention per sequence of `T` rows, fused | `scaled_dot_product_attention` |
| `g.Dropout(x, p)`, `g.Training` | dropout from the graph's own `g.Rng`, off when `Training` is false | `nn.Dropout`, `.train()` |
| a negative class | a padding row the loss ignores, the mean over the rest | `ignore_index` |
| `nn.Graph<F32>(T, 1, true)`, `g.Pos` | a graph for decoding: forward only, a key-value cache per attention | `past_key_values` |
| `g.Save(path)`, `g.Load(path)` | the parameter region to and from a file | `state_dict` |
| `g.Profiling`, `g.Times` | time per kind of operation, forward and backward | `torch.profiler` |

`Product` and `Multiply` are the builders of `MatMul` and `Mul`: olang reserves the operator methods' names (`MatMul`
is `@`, `Mul` is `*`) for every method of every type, so a graph cannot have methods called that (repro/operatornames).

- **A shorter batch** (an epoch's last) runs on the first `n` rows of every batched node - views, nothing re-planned;
  losses average over `n`. With attention, `n` is whole sequences (a multiple of `T`).
- **Gradients are written, then added to.** A `written` flag per node, reset each `Backward`, makes an op's first
  contribution to an input's gradient a write (`beta = 0` in a GEMM) and the rest additions (`beta = 1`), so no
  zero-fill pass runs. A parameter no path from the loss reaches is zero-filled; with `accumulate` set, parameters'
  gradients are added to instead (micro-batches).
- **An absent gradient is an empty matrix.** A kernel's backward takes every input's gradient destination and skips
  one with no rows (`if da.Rows > 0`) - an input, the classes or a constant has none, so the first layer computes no
  `dx`. (A nullable `Matrix&` would have meant a reference built per call.)
- **What needs a gradient** is decided at `Plan`: a node depending on a parameter (none in a graph for decoding).
- **Dropout is reproducible.** `Plan` seeds the graph's own generator `g.Rng` from the one it is given (after the
  parameters, so their starting values do not change); a mask is drawn by each training `Forward` and kept for the
  backward. Restoring `g.Rng` before a `Forward` replays a mask, which is how the gradient check checks dropout.

### The arena

One `Array<T>` per graph, laid out at `Plan`:

1. **parameters**, contiguous in registration order - the optimizers' one flat array, and a checkpoint;
2. **parameter gradients**, the same layout - one flat pass for the optimizer step and for gradient clipping;
3. every other node's **value**, its **gradient** when it needs one, and what its backward keeps (softmax cross-entropy
   its probabilities, the normalizations each row's statistics, attention its probabilities - `T x T` per sequence and
   head - and dropout its mask; in a graph for decoding, attention's key-value cache);
4. the products' **packing workspace** (`kernels.Gemm`, sized once from the graph's largest dimension) and attention's
   backward **scratch** (one `T x T` matrix per task).

Class indices live in an `Array<I32>` beside it; optimizer state (AdamW's `m` and `v`, the projection's workspace) is
made once, by the optimizer, in the parameters' layout. Liveness-based reuse of the activation region (a gradient is
dead once its producer's backward has run) is a planner change, made when a model's memory needs it.

## 3. Operations (`ops.olang`)

The kernels are plain functions on `linalg.Matrix<T>&`, forward and backward per primitive, usable without a graph -
which is what an eager mode would call too.

| op | forward | backward |
|---|---|---|
| `MatMul(a, b, ta, tb)` | `op(a) op(b)`, one GEMM | one GEMM per input, transposes swapped |
| `Linear(x, w, b)` | `x w^T`, then the bias row | `dx = dy w`, `dw = dy^T x`, `db` = column sums |
| `AddRow(x, b)` | every row plus `b` | `dx = dy`, `db` = column sums of `dy` |
| `Add`, `Sub`, `Mul` | element by element, same shape | element by element |
| `Scale(x, k)` | `k x` | `k dy` |
| `Relu`, `LeakyRelu`, `Gelu` (erf, PyTorch's default), `Sigmoid`, `Tanh` | one `Map` pass (`FastSigmoid`/`FastTanh`) | one pass on the saved input or output |
| `Softmax` | row-wise, max subtracted | `y (dy - rowsum(dy y))` |
| `SoftmaxCrossEntropy(logits, classes)` | mean over rows of `log-sum-exp - z[class]`; probabilities saved | `(p - onehot) / n`, fused |
| `Mse(a, b)` | mean of squared differences | `2 (a - b) / count` |
| `GeluTanh` | GPT-2's GELU, `0.5 x (1 + tanh(c (x + 0.044715 x^3)))`, through `FastTanh` | one pass on the input |
| `Embedding(tokens, table)` | gather: row `r` is the table's row `tokens[r]` | scatter-add into the table's gradient |
| `Positions(x, table, T)` | `x` plus the table's row `(Pos + r) mod T` | `dx = dy`; scatter-add by position |
| `LayerNorm(x, g, b)` | per row `(x - mean) rstd g + b`; mean and rstd saved | `rstd (dy g - mean(dy g) - xhat mean(dy g xhat))`, column sums |
| `RmsNorm(x, g)` | per row `x rinv g`; `rinv` saved | `rinv (dy g - xn mean(dy g xn))`, column sums |
| `Attention(q, k, v, heads, T, causal)` | per sequence and head `softmax(Q K^T / sqrt(dh)) V`, probabilities saved | `dQ = dS K`, `dK = dS^T Q`, `dV = P^T dO` |
| `Dropout(x, p)` | `x mask`, mask `1/(1-p)` or 0, kept | `dy mask` |

Every backward is checked against central differences in `F64` (`nn.olang`'s tests: each op through a small graph,
step 1e-6, agreement to 1e-6, the relative error floored at 1e-3 so a gradient near zero is compared absolutely - its
difference's own rounding is about 1e-16 x loss / h; a deliberately broken backward fails them). Fusion is the
planner's job, later: `Linear` then an activation as one GEMM with an epilogue, needing a `Gemm` with an epilogue in
std/linalg (section 13).

**The products go through `kernels.Gemm`**, std/linalg's algorithm with its packing panels in the graph's workspace -
a stopgap for the workspace `Gemm` std/linalg lacks (section 8, memory). **Attention's products are written as dot
products and updates** on each head's strided view (`kernels.Dot`, `kernels.Axpy`): a head's products are small (`T x
T x dh`), and these loops compute only the causal half - measured faster than a packed `Gemm` per sequence and head
(section 11).

## 4. Layers (`layers.olang`)

A layer is **plain data** - the handles of its parameters - made by a function that registers them on a graph, with
an `Apply` that records its ops:

```olang
type Dense struct(W nn.Var, B nn.Var, In I64, Out I64) { W  B  In  Out }

fn NewDense(g mut nn.Graph<<T>>&, inputs I64, outputs I64) Dense {
    w := g.Param(outputs, inputs, fanIn(inputs), true)      # weight decay on W, not on B
    b := g.Param(1, outputs, fanIn(inputs), false)
    return Dense(w, b, inputs, outputs)
}

fn (d Dense&) Apply(g mut nn.Graph<<T>>&, x nn.Var) nn.Var { return g.Linear(x, d.W, d.B) }
```

`NewMlp(g, I64[784, 128, 10], Activation.Relu)` holds a `List<Dense>` and applies them with the activation between.
Applying a layer twice shares its parameters. Starting values follow PyTorch's `nn.Linear`: weights and biases uniform
in `+-1/sqrt(in)`, drawn from the `rand.Rand` given to `Plan`, so a run is reproducible from its seed.

For transformers (section 11): `Norm` (layer or RMS normalization's gain and bias), `AttentionBlock` and
`Transformer`, built by `NewNorm`, `NewAttentionBlock(g, D, heads, T, layers, dropout, rms)` and `NewTransformer(g, V,
T, D, heads, layers, tied, dropout, rms)`, with GPT-2's starting values (`NewDenseWith` takes an `Init` for the
weights and one for the bias).

Why a factory function and not a constructor taking the graph: olang's zero values (D13c) are a constructor run on
zeros, and a constructor reading through a reference parameter has none, so such a struct can never be a `List`
element (repro/listzero). A plain-data layer has no such problem, and is serializable for free.

## 5. Losses and optimizers (`optim.olang`)

Losses are graph ops giving a `1 x 1` node: `SoftmaxCrossEntropy` (class indices, mean - PyTorch's
`CrossEntropyLoss`) and `Mse`; `g.Scalar(loss)` reads the number.

Optimizers update the parameter region **as one flat array in one fused pass** (PyTorch's fused optimizers), with
their state laid out the same way and made once. How weights are kept from growing is a `Regularizer`, so both
optimizers run with either:

```olang
type Regularizer enum {
    WeightDecay(rate F64)                  # decoupled: p -= lr * rate * p, on the parameters registered with decay
    Projection(alpha F64, normalized Bool) # the delta rule's input projection, in place of weight decay
}
```

- **`AdamW<T>(g, rate = 0.001, Beta1 = 0.9, Beta2 = 0.999, Eps = 1e-8, Reg = WeightDecay(0.01))`**, PyTorch's
  semantics: `m <- b1 m + (1 - b1) g`, `v <- b2 v + (1 - b2) g^2`, decoupled decay, then
  `p -= lr (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)` - one loop, `p = keep p - step m / (sqrt(v) root + eps)`
  with the bias corrections folded into two scalars. `Rate` may change between steps (a schedule).
- **`AdamWProjected(g, rate = 0.001, alpha = 1e-3, normalized = true)`** - AdamW with `Projection` in place of weight
  decay (Behrouz et al., "Nested Learning: The Illusion of Deep Learning Architectures"): for every dense layer
  `y = x W^T + b`, `W <- W (I - alpha x x^T) - lr * (Adam update)`, `x` the layer's **input**. For a batch of `n`
  rows that is `W <- W - (alpha / n) (W X^T) X`, and `W X^T` is the transpose of `M = Y - b`, the layer's output less
  its bias, which the forward pass already computed: so the projection is a pass over `Y - b` and **one** GEMM per
  dense layer (`Gemm(W, M, transA = true, X, false, -alpha / n, beta = 1)`), run before the Adam step changes `W`. It
  replaces weight decay. `normalized` divides each row's term by `|x|^2`, so alpha is the fraction of each weight row's
  component along an input's direction removed per step, whatever the inputs' scale - the default (the user's call,
  2026-10-09), because one alpha then fits every layer: unnormalized, alpha must shrink with `|x|^2` (1e-3 already
  costs MNIST five points, below).
- **`Sgd<T>(g, rate = 0.01, Momentum = 0, Nesterov = false, Reg = WeightDecay(0))`**, PyTorch's semantics (decay added
  to the gradient, momentum buffer, Nesterov's look-ahead).

The tests check AdamW's first steps against a hand computation, SGD's momentum, and the projection's formula against
its definition, on a small `F64` graph.

**Schedules and clipping.** `WarmupCosine(peak, warmup, total, floor)` gives a function value - the rate at each step:
linear up to `peak` over `warmup` steps, half a cosine down to `floor` at `total`, `floor` after (GPT-3's, as nanoGPT
has it). The `Optimizer` trait gained `SetRate(rate)`, so a loop sets the step's rate and an optimizer knows nothing of
schedules. `ClipGradNorm(g, max)` scales every gradient by one factor so their norm is at most `max` (PyTorch's
`clip_grad_norm_`) and gives the norm before - summed in `F64`: a long `F32` sum of small squares drops their low bits
and came out 3.5e-5 low against numpy.

## 6. Training and evaluation (`train.olang`, `examples/mnist_mlp.olang`)

```olang
data := try mnist.Load("data/mnist", mnist.Source)
g := nn.Graph<F32>(128, threads)
x := g.Input(784)
y := g.Classes()
net := layers.NewMlp(g, I64[784, 128, 10])
logits := net.Apply(g, x)
loss := g.SoftmaxCrossEntropy(logits, y)
r := rand.Rand(seed)
g.Plan(r)
c := train.Classifier(x, y, logits, loss)
o optim.AdamW<F32>&g = optim.AdamW<F32>(g)
trainSet := loader.Loader(data.Train, 128, seed)
testSet := loader.Loader(data.Test, 128, seed, false)
for e in range 1, epochs + 1 {
    trainLoss := train.Epoch(g, c, trainSet, o)         # Fill, Forward, Backward, Step per batch
    testLoss, accuracy := train.Evaluate(g, c, testSet) # Forward per batch, ArgMaxRows against the classes
}
```

`make mnist ARGS="10 adamw 1 1"` runs it: epochs, optimizer (`adamw`, `projected`, `sgd`), seed, threads, and for
`projected` its alpha and `1` to normalize.

A language model's loop (`train.LanguageModel`, `train.Gpt`, `LmStep`, `LmEvaluate`, `examples/charlm.olang`):

```olang
c := try text.Load("data/shakespeare/input.txt")
g := nn.Graph<F32>(16 * 64, threads)                      # 16 sequences of 64 characters
m := train.Gpt(g, c.Size(), 64, 128, 4, 4)                # vocabulary, context, width, heads, layers
g.Plan(r)
o optim.AdamW<F32>&g = optim.AdamW<F32>(g, 1e-3, 0.9, 0.99, 1e-8, optim.Regularizer.WeightDecay(0.1))
rate := optim.WarmupCosine(1e-3, 100, steps, 1e-4)
windows := text.Windows(c.Train(), 64, seed)
for step in range steps {
    loss, norm := train.LmStep(g, m, windows, o, rate(step), 1.0)   # random windows, clip to 1, a step
}
val := train.LmEvaluate(g, m, text.Windows(c.Val(), 64), 10)     # the same windows every time, dropout off
```

`make charlm ARGS="2000 1 2"` trains it (steps, seed, threads, dropout, characters to generate), saves
`data/shakespeare/charlm.ckpt` and generates a sample; `examples/charlm_sample.olang` loads the checkpoint and
generates with and without the key-value cache.

## 7. Datasets (`datasets/`)

- `datasets/idx` reads the IDX format; its elements are a view of the file's bytes.
- `datasets/loader`: `Labeled` holds a set as bytes with a per-feature `Scale` and `Shift` (default `1/255` and `0`).
  `Loader` makes a seeded order each epoch (Fisher-Yates, std/rand's xoshiro256**) and `Fill(b, x, y)` writes batch
  `b` as F32 rows and classes into the graph's own buffers, giving its size (the last is short). Nothing is allocated
  per batch.
- `datasets/mnist` fetches the four files with curl and gunzip into `data/mnist` (each written to a `.part` file and
  renamed when whole), validates them and loads both sets. Fashion-MNIST and KMNIST are the same call with another
  address.
- The images stay bytes (47MB, against 188MB as F32). The four files load at 520-830MB/s and filling shuffled batches
  produces ~3.5-4GB/s of F32 (~1.2M samples/s) - 3% of an epoch.

- `datasets/text`: a `Corpus` of characters - its vocabulary (the distinct bytes, in byte order, so a text always gives
  the same ids), the text as `I32` tokens, nanoGPT's 90/10 split into `Train()` and `Val()` - and `Windows` over a
  token sequence: `Random` fills a batch of `B` windows of `T + 1` tokens at uniform offsets (nanoGPT's `get_batch`),
  `At` the windows of an ordered pass, the inputs into the graph's tokens and each one's next token into its classes.
  `Fetch` gets tiny Shakespeare with curl (written to a `.part` file and renamed).

Next: a `Dataset` trait (a constraint) so one `Loader<D>` takes any source, a tokenizer beyond characters (byte-pair
encoding), and filling the next batch on a task while the step runs.

## 8. Results and performance (measured 2026-10-09)

**Accuracy**, 784-128-10, ReLU, batches of 128, 10 epochs, test set:

| optimizer | test accuracy |
|---|---|
| AdamW, lr 1e-3, decay 0.01 | 97.73% (seed 1), 97.71% (seed 2), 97.80% (seed 3); above 97% from epoch 5 |
| SGD, lr 0.05, momentum 0.9 | 97.81% |
| AdamW + projection, alpha 1e-5 | 97.74%, 97.53%, 97.84% (seeds 1-3) |
| AdamW + projection, alpha 1e-4 / 1e-3 | 97.17% / 92.52% (too strong) |
| AdamW + projection, normalized, alpha 1e-3 / 1e-2 | 97.79% / 97.74% |

**The C reference** (`bench/ref/mlp.c`): the same network in C over OpenBLAS - same random numbers (xoshiro256**
seeded by splitmix64), same initialization order, shuffle, scaling and AdamW - so its losses match oann's to four
decimals in the first epoch, and the time is the comparison. PyTorch could not be installed: download.pytorch.org is
refused by the container's network policy (403).

**An epoch** (469 steps), seconds, medians of interleaved rounds (`make epoch`), on the shared 4-core machine at a
load average of 4-5 (other jobs running, which is also why 4 threads lose):

| | 1 thread | 4 threads |
|---|---|---|
| oann (olang, default target) | 2.39 | 2.90 |
| oann, its IR relinked with `-march=native` | 1.92 | - |
| C over OpenBLAS | 0.79 | 1.19 |

Where oann's epoch goes (`bench/train.olang`): forward 43%, backward 50% - both nearly all GEMM - filling batches 3%,
the optimizer 4%. The gap to OpenBLAS is therefore std/linalg's GEMM, and it is the instruction set: olang builds for
baseline x86-64 (SSE2) and never contracts `a*b + c`, while OpenBLAS runs AVX-512 with FMA; std/linalg's GEMM is level
with the same algorithm in C at that target. Closing it is a compiler direction (a native target, FMA contraction),
not an oann one. The projection adds 0.6-1s an epoch (one GEMM per dense layer per step). Results are identical for
every thread count (GEMM partitions its output).

**Memory - a leak, not oann's, now worked around.** A step allocates nothing of its own (checked: 200,000 forward
passes of a graph whose products are not packed grow by nothing). But std/linalg's `Gemm` packs into two scratch arrays
made in its own scope per call, and the runtime's chunk pool reuses only its head chunk: the B panel's chunk, taken
first, ends up behind the A panel's and is never reused, so every packed product maps a new one. MNIST training grew
~540KB a step, ~215MB an epoch (2.2GB over 10 epochs); the transformer of section 11, with a few hundred products a
step, grew **8.1MB a step** (243.6MB over 30 steps - a 2,000-step run would have taken 16GB). Reproducer:
`repro/chunkpool.olang` (no linalg - two scratch arrays, large then small). **Worked around in phase 3:**
`kernels.Gemm` is std/linalg's algorithm with its panels in the graph's arena, and the same transformer grows by 72kB
over the first 30 steps and nothing after - a step allocates nothing at all. Fixes still wanted upstream: first-fit in
the runtime's pool, and a `Gemm` taking a workspace in std/linalg (section 13).

## 9. Testing

- `make test`: each module's `test` blocks - the kernels against direct computations, every op's backward against
  central differences in `F64` (a graph per op, a batched network on a short batch with fan-out, accumulation), the
  layers' registration and initialization, the optimizers against hand computations, and the datasets.
- `make data`: the MNIST pipeline against what is known about the dataset (counts, first labels, class histograms,
  pixel mean 0.1307 and deviation 0.3081), timed.
- Transformers (phase 3): every new backward against central differences in `F64` - embedding (with repeated tokens),
  positions, both normalizations, the tied head, padding rows (`nn.olang`); attention causal and not, two and four
  heads, a node attending to itself as q, k and v, on full and short batches; dropout with a fixed mask; GELU through
  tanh; a whole two-layer transformer tied and untied, with dropout and with RMS normalization (`layers.olang` - a
  deep network's own central-difference error bottoms out at 1e-6 to 4e-6, so its bound is 1e-5); causality (a row's
  output does not change with a later row); the key-value cache against running the whole sequence (`generate.olang`,
  to 1e-12); the workspace `Gemm` against std/linalg's for every transpose, beta, size and thread count
  (`kernels.olang`); the schedule, clipping, checkpoints and the text corpus.
- `make lmref`: the first steps of the character model - losses and gradient norms - against the same model, the same
  starting values, windows, AdamW, schedule and clipping in numpy (`bench/ref/charlm.py`, F64).
- `make lmbench`: where a transformer's step goes, by kind of operation (the graph's own profiling).
- `make mnist` trains end to end (needs the downloaded data); `make epoch` times it against C.

## 10. Serialization (later)

Checkpoints as **safetensors** (an 8-byte header length, a JSON header naming each matrix's dtype, shape and byte
range, then the raw little-endian data): interoperable with PyTorch and Hugging Face, written with std/json and the
floats' `Bits()`; the parameter region is already one contiguous block. Parameters carry PyTorch's names
(`l1.weight`, `l1.bias`).

## 11. Transformers (phase 3)

A decoder-only transformer (GPT-2's) trained on a token stream, with no tensor type:

- **Rows are tokens.** A batch of `B` sequences of `T` tokens is a `[B*T, D]` matrix, sequence `s` in the `T` rows from
  `s T`; the graph is planned for `Batch = B*T` rows, and a node knows `T` where it needs it (attention, positions).
  The loss is `SoftmaxCrossEntropy` over the `B*T` rows of vocabulary logits, the classes each token's next one.
- **Leaves and ops**: `Tokens()` (an `I32` per row, as `Classes()`); `Embedding(tokens, table)` (a gather; the
  backward scatter-adds into the table's gradient, so a token repeated in a batch sums); `Positions(x, table)` (learned:
  row `(Pos + r) mod T` of a `[T, D]` parameter); `LayerNorm(x, gain, bias)` and `RmsNorm(x, gain)` (row-wise, `eps`
  1e-5, the biased variance as PyTorch has it, each row's statistics saved); `Attention(q, k, v, heads, T, causal)`;
  `Dropout(x, p)` (inverted, from the graph's own generator, off when `g.Training` is false); `GeluTanh` (GPT-2's
  GELU); a negative class marks a padding row the loss ignores (PyTorch's `ignore_index`: the mean is over the rows
  counted, and a batch of padding alone has a loss of zero).
- **Attention is one fused op.** Head `h` is columns `h D / heads ...` of `q`, `k`, `v` - a strided view, nothing
  copied - and per sequence and head `P = softmax(Q K^T / sqrt(dh))` (row `i` over positions `0 ... i` when causal),
  `Y = P V`. The probabilities are saved for the backward (`T x T` per sequence and head - the memory to watch at long
  contexts; a tiled, recomputing form is the answer then). The backward: `dS = P (dO V^T - rowsum(dO V^T P)) / sqrt(dh)`,
  then `dQ = dS K`, `dK = dS^T Q`, `dV = P^T dO`, written in that order - the order their "written" flags were taken
  in - so a node attending to itself (`q`, `k` or `v` the same node) adds its three contributions up. Sequences are
  spread over the graph's threads; each task has its own `T x T` scratch in the arena. The products are dot products and
  updates over the heads' views (`kernels.Dot`, `kernels.Axpy`) rather than a `Gemm` per sequence and head: at these
  sizes packing costs more than it saves, and the loops compute only the causal half (`bench/attention.olang`: 4.4 ms
  against 6.6 forward at B 16, T 64, D 128, 4 heads; 21.6 against 27.6 at T 256).
- **Layers** (`layers.olang`): `NewTransformer(g, V, T, D, heads, layers, tied = true, dropout = 0, rms = false)` - a
  token embedding plus learned positions (dropout after), `layers` pre-normalized blocks, a final normalization and the
  logits against the embedding itself when tied (weight tying is applying one parameter twice: the head's product and
  the embedding's scatter-add both add into its gradient), a `V x D` matrix of its own otherwise. `NewAttentionBlock`:
  `h = x + O(attention(Q n, K n, V n))`, `n = norm1(x)`; `y = h + Down(gelu(Up(norm2(h))))`, `Up` 4D wide, dropout on
  both branches. GPT-2's starting values (weights and embeddings `N(0, 0.02)`, the two projections into the residual
  stream `N(0, 0.02 / sqrt(2 layers))`, biases 0, gains 1) and nanoGPT's decay groups (weights and embeddings decayed,
  biases and gains not). `q`, `k` and `v` are three products, not one of width 3D: that needs view nodes (section 14).
- **GELU through tanh, not erf**, in the blocks: the exact GELU's `erf` is a library call per element - 14-25 ns an
  element, 62 ms of a 510 ms step - where the tanh form (GPT-2's own) runs through `FastTanh` in one pass at 5-8 ns.
  Both are ops; the tanh form is within 1e-3 of the exact one.
- **Training**: AdamW (beta2 0.99, decay 0.1), `optim.WarmupCosine` for the rate, `optim.ClipGradNorm` to 1.
- **Generation** (`generate.olang`): `Sample(logits, temperature, r)`; `Rerun` runs the last `T` tokens through a graph
  planned for one sequence for every new token; `Cached` runs each token once on a **graph for decoding**
  (`nn.Graph<F32>(T, 1, true)`): no gradients, and every attention keeps the keys and values of the positions so far
  in its saved region - a key-value cache in the arena - so `Forward(n)` at `g.Pos` computes the `n` positions from
  `Pos`, each attending to everything cached, `Positions` adding row `Pos + r`. The prompt is one `Forward` (a prefill); when
  the context is full, the last `T / 2` tokens are run again to start a fresh cache. Checked: decoding a prefill and
  then token by token gives the logits of running the whole sequence, to 1e-12.
- **Checkpoints**: `g.Save(path)`/`g.Load(path)` - the parameter region as it is (`oann`, the element width and the
  count, then little-endian bits); safetensors when a model has to leave oann (section 10).

### Validation

Every new backward agrees with central differences in `F64` (section 9). Against a reference: `bench/ref/charlm.py`
is the character model again in numpy, F64, with a hand-written backward, started from the **same parameters** -
xoshiro256** and splitmix64 in Python, the same draws in the same order, rounded to F32 as oann stores them - and fed
the **same windows**, with the same AdamW, schedule and clipping. The first ten steps (`make lmref`), oann in F32
against numpy in F64:

| step | loss, oann | loss, numpy | gradient norm, oann | gradient norm, numpy |
|---|---|---|---|---|
| 1 | 4.211253 | 4.211253 | 7.050046 | 7.050046 |
| 2 | 4.181737 | 4.181737 | 6.326455 | 6.326455 |
| 5 | 4.033323 | 4.033323 | 5.406032 | 5.406031 |
| 10 | 3.751469 | 3.751469 | 1.995772 | 1.995772 |
| 20 | 3.552264 | 3.552264 | 1.526185 | 1.526185 |
| 40 | 3.034891 | 3.034891 | 1.233417 | 1.233417 |

Over 40 steps the losses differ by at most 1e-6 and the norms by 2e-6 - what F32 against F64 arithmetic allows - so
the forward, every backward, AdamW, the schedule and the clipping are the reference's. One thing this found: the
gradient norm summed in `F32` came out 3.5e-5 low (a long sum of small squares drops their low bits), so
`ClipGradNorm` sums in `F64`. PyTorch could not be installed (the proxy refuses pip and download.pytorch.org); numpy
2.5 was there.

### Results: a character-level model on tiny Shakespeare

`make charlm ARGS="2000 1 2"`: 4 layers of width 128, 4 heads, context 64, batches of 16 sequences (1,024 tokens),
tied embeddings, 809,856 parameters, dropout 0; AdamW at 1e-3 (beta2 0.99, decay 0.1) warmed up over 100 steps and down
a cosine to 1e-4 at step 2,000, the gradient clipped to norm 1; seed 1, 2 threads. tiny Shakespeare was fetched with
curl through the environment's proxy (1,115,394 characters, a vocabulary of 65; the first 90% trained on, the last 10%
validated on - the same 10 batches of 16 windows each time). Losses in nats per character:

| step | train (mean of the last 50 steps) | validation |
|---|---|---|
| 0 | - | 4.2086 (ln 65 = 4.174) |
| 250 | 2.4134 | 2.3825 |
| 500 | 2.1998 | 2.2186 |
| 750 | 2.0541 | 2.0828 |
| 1000 | 1.9224 | 2.0032 |
| 1250 | 1.8402 | 1.9575 |
| 1500 | 1.7713 | 1.8779 |
| 1750 | 1.7041 | 1.8220 |
| 2000 | 1.7023 | 1.8063 |

2,000 steps (2M tokens, about two passes over the training text) took 1,108 s - 554 ms a step, **1,849 tokens a
second** - on the shared 4-core machine at a load average of 6 to 9 (other agents compiling and testing). Still falling
when the schedule ended: a longer run, a larger model and dropout are the obvious next steps (nanoGPT's 10.7M-parameter
"baby GPT" reaches 1.47 after 5,000 steps of 16K tokens).

A sample (`examples/charlm_sample.olang`: 500 characters after a newline, temperature 0.8, the key-value cache):

```
Pears Richmpy'd, had lies his are pinter and morige:
Where the with my lordiety hims thee flown.

NORGEOs:
Hard, poor night inter a clird it.

Secourrongel:
There as contranted you the fair him to liash,
The pring sweets he place on murd, if hor know you sins,
Bold in with blood and my streen have thou
with no she be behose farbels fear the men with preoving to bid
Not for a son the see that she fintlew on shoul,
Disss of the relikess, thou rady now the load the Glotheres,
I'll pord to my foul m
```

Generation: **2,138 characters a second with the key-value cache, 92 running the context again** (23x; one thread,
at a load of 3). Greedily the two give the same 58 characters while the context is not full; after that they differ
by design (the cache restarts from the last 32 tokens, the rerun keeps the last 64).

### Where a step goes

`make lmbench ARGS="20 1"` - the graph's own profiling, ms a step, one thread, at a load average of 2.6:

| | forward | backward |
|---|---|---|
| `Linear` (12 products a step forward, 24 backward) | 138.5 | 281.5 |
| `MatMul` (the tied head) | 1.5 | 3.0 |
| `Attention` | 20.3 | 24.7 |
| `GeluTanh` | 12.5 | 14.9 |
| `LayerNorm` | 2.9 | 4.2 |
| `Add` (residuals) | 1.3 | 1.8 |
| the loss, embedding, positions | 0.8 | 0.3 |
| clipping, AdamW | 4.0 (together) | |

A step is **512 ms: products 83%, attention 9%, the rest of the graph 8%**, clipping and the optimizer under 1%. The
products run at ~11 GFLOPS (4.8 GFLOP a step) - std/linalg's GEMM on the default SSE2 target, slower than its 14-17 on
large squares because these are narrow (k = 128) - so the step is the compiler's instruction set, as MNIST's epoch was
(section 8). Two threads: 399 ms (2,566 tokens a second), four: 344 ms (2,977) - the products scale (424, 326, 275
ms), the element-wise ops do not (they run on one thread). The arena is 87.2 MB, laid out once; resident memory grows
by 70-84 kB over the first steps and by nothing after (100 steps measured). MNIST's epoch with the workspace `Gemm`:
2.2 s (2.39 before), and it no longer grows.

### olang issues found on the way (repro/)

- `repro/chunkpool.olang` (phase 2) - std/linalg's `Gemm` leaks a chunk per packed product through the runtime's pool;
  for a transformer 8.1 MB a step. Worked around by `kernels.Gemm`.
- `repro/capturedfn.olang` - a lambda that captures a function value and calls it (`fn(d, g, v) { return d + f(g, v)
  }`) leaves an indirect call per element even when everything is inlined: 3.1 ns an element against 0.55 written
  out. `ops.backward2` now runs its loops itself.
- `repro/ctorunstored.olang` - O26 counts an instance as referring to every reference its constructor was given, even
  one it only reads: `return Counts(t)` with `t` a local is refused. `text.Load` declares its text `&return` instead.
- Not issues, recorded for the language's records: `:=` from a comparison (`ta := t % 2 == 1`) still needs its type
  written (D15 - being relaxed); a `match` value cannot give several results (`=> rows, cols`), so `savedShape` uses
  statements; a text join's piece cannot be a conditional (`$` of a local holding it is).

### What remains

- **Mixed precision** - BF16 storage with F32 master weights - not started: the graph is generic over one element type
  (section 14), and std/linalg's BF16 product allocates per call (the leak) and widens while packing.
- Dropout of attention's probabilities (nanoGPT's `attn_dropout`): only the residual branches and the embedding are
  dropped out.
- A tiled attention that recomputes `P` in the backward instead of saving `T x T` per head (memory at long contexts),
  and a batched causal `Gemm` under it (section 13).
- Fused QKV and splitting heads with view nodes; a `Gemm` epilogue for bias and GELU (section 13).
- Rotary positions; a byte-pair tokenizer; generating several sequences at once (a decoding graph of `B` sequences).
- Reusing activation storage by liveness: the arena holds every activation and gradient - 87 MB for the model above.
- Element-wise ops over threads: at four threads they are 40 of the 344 ms.

## 12. Shapes as type parameters (when olang has const generics)

olang is getting `Matrix<T, R, C>`, each dimension a constant or known at run time. oann is ready for it: a layer's
shapes are given once, at construction, and `Apply` takes only handles, so `Dense<T, In, Out>` can hold
`W Matrix<T, Out, In>` and a handle carry its column count (`Var<C>`), making a shape mismatch in a model a
compile-time error. The executor keeps run-time shapes, as JAX keeps shapes as data while checking them when tracing.

## 13. What oann needs from std/linalg next

oann does all its arithmetic through `linalg.Matrix<T>` - `View`, `Row`, `Gemm`, `Map`/`Map2`/`Map3`, `AddRow`,
`ColumnSums(out, beta)`, `RowSoftmax`, `RowNorms`, `ArgMaxRows`, `Fill`, `FillUniform`/`FillNormal`, `Cast`, the
`Fast*` functions. What it still needs:

1. **`Gemm` into a caller's workspace**, so a planned graph gives the packing panels a place in its arena and a step
   allocates nothing at all - and the leak in section 8 cannot happen whatever the runtime does. **Needed**: without
   it a transformer's training grew 8.1MB a step; oann now carries a copy of std/linalg's algorithm with a workspace
   (`kernels.olang`, ~250 lines) that should go when this exists. Wanted shape: `GemmSpace(m, n, k, threads)` and
   `Gemm(c, a, ta, b, tb, alpha, beta, threads, work)`.
2. **Batched strided `Gemm`, causal-aware** - every sequence and head of attention in one call, the products of a
   lower-triangular block skipped. Attention is 9% of a step at context 64 and grows as `T^2`; oann's dot-product
   kernels run it at 3-4 GFLOPS on the causal half, against ~12 for `Gemm` on large products, and a `Gemm` per head
   is slower still (6.6 ms against 4.4 forward at B 16, T 64; 27.6 against 21.6 at T 256 - packing costs more than
   it saves at these sizes). A batched causal `Gemm` at the products' ~11 GFLOPS would run these about 3x faster -
   ~30 of attention's 45 ms a step at T 64 - and matters more at longer contexts, where attention's share grows.
3. **`Gemm` with an epilogue** - `c = act(alpha op(a) op(b) + bias row)` - so `Linear` plus an activation is one pass:
   the bias row and GELU are two more passes over the widest activation (B T x 4D) - GELU's forward alone is 12.5 ms of
   a 512 ms step, its backward 14.9 - so ~5%; and
   **fused QKV** (one product of width 3D, attention reading q, k and v as column blocks) needs view nodes in oann's
   graph, not linalg.
4. **Micro-kernels for the machine** (AVX2/AVX-512 with FMA - std/linalg's per-CPU kernels, with the compiler's
   native target): the products are 83% of a transformer's step at ~11 GFLOPS, and OpenBLAS ran MNIST's 3x faster by
   its instruction set alone (section 8) - the largest lever on a step, ~2x or more. The same target is what
   vectorizes `FastTanh` cheaply (~5 ns an element on SSE2: GELU's 27 ms a step).
5. **Two-operand indexing**, `m[r, c]` - now in olang (E31 multi-index), not yet used by oann.

## 14. Open questions

1. The projection: normalized, alpha 1e-3 by default (97.79% on MNIST, section 8); still open whether alpha should
   scale with the learning rate and whether it should apply to every dense layer or only some.
2. The GEMM gap to OpenBLAS (3x single-threaded) is the target instruction set; a native-target build mode (and FMA
   contraction) in the compiler would close most of it.
3. The chunk-pool leak (section 8): worked around in oann by `kernels.Gemm`; upstream, first-fit in the runtime, a
   workspace `Gemm` in std/linalg, or both - and then `kernels.olang`'s copy goes.
4. An eager mode beside the graph, for models whose structure depends on their data - not before a model needs one.
5. oann as an importable package (`github.com/OWNER/oann/nn`) - the layout already is one module per concern.
6. Mixed precision (BF16 storage, F32 master weights): the graph is generic over one element type, so it needs either
   a second arena of BF16 copies the products read (and std/linalg's BF16 product into a workspace - today it
   allocates per call and widens while packing) or a graph with a type per node. Not started (section 11).
7. View nodes - a node whose value is a column block of another's, no storage of its own - would give fused QKV (one
   product of width 3D) and splitting heads at no cost; the "written" flag per node would then need to be per block.

## 15. Layout

```
makefile                OLANG ?= the compiler; make test, data, mnist, epoch, charlm, lmbench, lmref, clean
nn.olang                Graph, Var, Op, Node: recording, Plan, Forward, Backward, decoding, checkpoints,
                        profiling; the gradient checks
ops.olang               the kernels: forward and backward per primitive, on Matrix
kernels.olang           Gemm into a workspace (std/linalg's algorithm), Dot, Axpy, Sum, SquaresF64
layers.olang            Dense, Mlp, activations, PyTorch's initialization; Norm, AttentionBlock, Transformer
optim.olang             the Optimizer trait, Regularizer, AdamW (and AdamWProjected), Sgd; WarmupCosine, ClipGradNorm
train.olang             Classifier, Epoch, Evaluate; LanguageModel, Gpt, LmStep, LmEvaluate
generate.olang          sampling; generation running the context again, and with a key-value cache
datasets/idx.olang      the IDX format
datasets/loader.olang   Labeled sets and the Loader
datasets/mnist.olang    fetching and loading MNIST
datasets/text.olang     a character corpus, its split, and windows of tokens; fetching tiny Shakespeare
examples/mnist_mlp.olang
examples/charlm.olang   the character-level transformer on tiny Shakespeare, trained, saved and sampled
examples/charlm_sample.olang  sampling from its checkpoint, with and without the cache
bench/data.olang        the data pipeline, checked and timed
bench/train.olang       where an epoch's time goes
bench/epoch.sh, ref/    an epoch against the C reference over OpenBLAS
bench/lm.olang          where a transformer's step goes, by kind of operation
bench/lmref.olang, ref/charlm.py  the first steps of the character model against numpy
bench/attention.olang   attention's kernels against a Gemm per sequence and head
docs/settling.md        settling networks - the model after transformers
repro/                  minimal programs for olang issues found here
data/, build/           downloads and build output, not in git
```

Each file is one olang module, imported by its path relative to the importing file without the extension
(`import "../datasets/mnist"`). Random numbers are std/rand's, times std/time's.
