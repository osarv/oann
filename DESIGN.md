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
  losses average over `n`.
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

## 11. Next: transformers

The graph is shaped for a decoder-only transformer (GPT-2 style) trained on a token stream, without a tensor type:

- **Rows are tokens.** A batch of `B` sequences of `T` tokens is a `[B*T, D]` matrix; the graph is planned for
  `Batch = B*T` rows, and a node knows `T` where it needs it (attention, positions). The loss is the existing
  `SoftmaxCrossEntropy` over `B*T` rows of vocabulary logits.
- **New leaves and ops**:
  - `Tokens()` - like `Classes()`, an `I32` per row;
  - `Embedding(tokens, table)` - gather rows of a `[V, D]` parameter; backward scatter-adds into the table's gradient;
  - `Positions(x, table, T)` - add row `t mod T` of a `[T, D]` parameter (learned positions); rotary positions as an
    option inside attention;
  - `LayerNorm(x, gain, bias)` and `RmsNorm(x, gain)` - row-wise, saving the row's mean and inverse deviation;
  - `Attention(q, k, v, heads, T, causal)` - **one fused op**: per sequence and head, `softmax(q_h k_h^T / sqrt(d) +
    mask) v_h`, heads being column blocks of `[B*T, D]` views (strided, no copies), each product a GEMM on those views;
    the probabilities saved for the backward (`T x T` per head and sequence - the memory to watch), a tiled
    (flash-style) form later for long sequences;
  - `Dropout(x, p)` - a mask from the graph's own `Rand`, active while `g.Training`;
  - `Gelu`, `Add` (residuals), `Linear` and `SoftmaxCrossEntropy` exist.
- **Layers**: `NewAttentionBlock(g, D, heads)` (layer norm, the `q`/`k`/`v` and output projections, the MLP with
  GELU, two residuals), `NewTransformer(g, V, T, D, heads, layers)`; weight tying of the embedding and the output
  projection is applying the same parameter twice, which the graph already supports.
- **Training**: AdamW with a warmup-then-cosine schedule (a function value giving the rate per step), gradient
  clipping by global norm over `g.ParamGrads()`, and BF16 storage with F32 master weights and accumulation once the
  F32 version matches a reference.
- **A new shape is a new plan**: sequences are padded to `T` with the padded tokens' loss masked (a `Constant` mask
  row), so one plan serves a run; generation, where `T` grows, uses a key-value cache in the graph's workspace.
- Validation as for MNIST: every new backward against central differences in `F64`, then a small character-level
  model trained against the same model in a reference.

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
   lower-triangular block skipped. Attention is 8% of a step at context 64 and grows as `T^2`; oann's dot-product
   kernels run it at 3-4 GFLOPS on the causal half, against ~12 for `Gemm` on large products, and a `Gemm` per head
   is slower still (6.6 ms against 4.4 forward at B 16, T 64; 27.6 against 21.6 at T 256 - packing costs more than
   it saves at these sizes). Expected: attention 2-3x faster, ~5% of a step at T 64, ~15-20% at T 256.
3. **`Gemm` with an epilogue** - `c = act(alpha op(a) op(b) + bias row)` - so `Linear` plus an activation is one pass:
   the bias row and GELU are two more passes over the widest activation (B T x 4D), ~15 ms of a 550 ms step; and
   **fused QKV** (one product of width 3D, attention reading q, k and v as column blocks) needs view nodes in oann's
   graph, not linalg.
4. **Vectorizing exp and tanh on the target** - `FastTanh` costs ~5 ns an element on the default SSE2 target, which
   keeps GELU at ~25 ms a step; the native target (AVX2, FMA) is the compiler's (pending).
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
