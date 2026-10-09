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
model in numpy to six decimals, and a character-level model trained on tiny Shakespeare. **Phase 4** (section 16):
a byte-level BPE tokenizer, and the transformer trained on its tokens; safetensors checkpoints; convolution and max
pooling (im2col and one product for the whole batch), and a small CNN on MNIST. **Settling networks** (section 17,
docs/settling.md): phases 1 and 2 - circuits that settle, learning by free and nudged phases, a spiking variant, and an
agent with memories and arousal. **Built with olang 9621af3** (section 8): code for this machine by default (AVX-512
with FMA here) and every product through std/linalg's `GemmWorkspace` - an MNIST epoch went from 2.39 s to 0.69, now
ahead of the C reference over OpenBLAS (0.93 s, interleaved), and a transformer step from 512 ms to ~200.

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
- **Structure beyond two dimensions lives in the operation.** A convolution reads each row as one `H x W x C` image,
  channels last (im2col into a workspace, one GEMM for the whole batch: section 16); attention reads rows as `B x T`
  tokens and its heads as column blocks (section 11).

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
  causal)`, `Dropout(x, p)`, `SoftmaxCrossEntropy(logits, classes)`, `Mse`, `Conv2d(x, w, b, shape)`,
  `MaxPool2d(x, pool)`, and the leaves `Input`, `Classes`, `Tokens`, `Constant`, `Param(init, decay)`.
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
| `g.Conv2d(x, w, b, inC, outC, k, stride, pad, H, W)`, `g.MaxPool2d(x, C, H, W, k, stride)` | images as rows, channels last | `nn.Conv2d`, `nn.MaxPool2d` on NHWC |
| a negative class | a padding row the loss ignores, the mean over the rest | `ignore_index` |
| `nn.Graph<F32>(T, 1, true)`, `g.Pos` | a graph for decoding: forward only, a key-value cache per attention | `past_key_values` |
| `g.Save(path)`, `g.Load(path)` | the parameter region to and from a file | `state_dict` |
| `checkpoint.Save(g, path, names, dtype)`, `checkpoint.Load(g, path, names)` | the parameters as safetensors, by name | `save_file`, `load_state_dict` |
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
4. attention's **scratch** (per task, a `T x T` matrix for the backward's `dS` and a `dh x T` one for a transposed
   block) and the convolutions' **patches** (one region, the largest convolution's im2col matrix, shared by all of
   them).

The products pack their operands into a `linalg.GemmWorkspace<T>` the graph keeps beside the arena: it grows, where
the graph lives, to the largest product during the first step, and allocates nothing after. Attention's products run
in tasks, so each task packs into a workspace of its own - one per thread, made by `Plan` and grown there, by running
attention's products once on zeros, so that no task ever allocates. Class indices live in an
`Array<I32>` beside it; optimizer state (AdamW's `m` and `v`, the projection's workspace) is
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
| `Conv2d(x, w, b, shape)` | im2col, then `cols w^T + b` - one GEMM for the batch (`conv.olang`) | `db` = column sums, `dw = dY^T cols` (patches rebuilt), `dcols = dY w`, col2im |
| `MaxPool2d(x, pool)` | each window's maximum per channel; where it was, kept | the gradient added where the maximum was |

Every backward is checked against central differences in `F64` (`nn.olang`'s tests: each op through a small graph,
step 1e-6, agreement to 1e-6, the relative error floored at 1e-3 so a gradient near zero is compared absolutely - its
difference's own rounding is about 1e-16 x loss / h; a deliberately broken backward fails them). Fusion is the
planner's job, later: `Linear` then an activation as one GEMM with an epilogue, needing a `Gemm` with an epilogue in
std/linalg (section 13).

**The products go through std/linalg**: `ws.Gemm(...)` on the graph's `GemmWorkspace` (section 2), so they run at
std/linalg's per-target tiles (AVX-512 with FMA on this machine) and a step allocates nothing. **Attention's products
are too** - a `Gemm` per sequence and head on the heads' strided views, the whole `T x T` square even when causal
(section 11). Only decoding with a key-value cache still runs dot products and updates over runs of arrays
(`kernels.Dot`, `kernels.Axpy`), one new row at a time.

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

For images (`vision.olang`, section 16): `Conv`, made by `NewConv(g, inC, outC, kernel, H, W, stride, padding)` with
PyTorch's Conv2d starting values. For transformers (section 11): `Norm` (layer or RMS normalization's gain and bias),
`AttentionBlock` and `Transformer`, built by `NewNorm`, `NewAttentionBlock(g, D, heads, T, layers, dropout, rms)` and
`NewTransformer(g, V, T, D, heads, layers, tied, dropout, rms)`, with GPT-2's starting values (`NewDenseWith` takes an
`Init` for the weights and one for the bias).

Why a factory function and not a constructor taking the graph: a layer holds only handles, so it is the same for
every element type, while the graph is generic (`Graph<T>`) - and a constructor of a non-generic type cannot take a
generic parameter: the declaration is accepted and every call fails (repro/genericctor). (The first reason recorded
here, that a constructor reading through a reference parameter left the struct no zero value and so no place in a
`List`, went with olang a3ed507.) A plain-data layer is also serializable for free.

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

Tokens beyond characters are `tokenizer.olang`'s byte-level BPE (section 16): its ids go into the same `Windows`.
Next: a `Dataset` trait (a constraint) so one `Loader<D>` takes any source, and filling the next batch on a task while
the step runs.

## 8. Results and performance (measured 2026-10-09)

**Accuracy**, 784-128-10, ReLU, batches of 128, 10 epochs, test set. First measured with olang ef939ae (baseline
x86-64); the rows run again with olang 9621af3 (this machine's AVX-512, its products' multiply-adds fused, so the last
bits of every product differ and a run drifts a little from the old one):

| optimizer | test accuracy, ef939ae | 9621af3 |
|---|---|---|
| AdamW, lr 1e-3, decay 0.01 | 97.73% (seed 1), 97.71% (seed 2), 97.80% (seed 3); above 97% from epoch 5 | 97.79%, 97.69%, 97.79% |
| SGD, lr 0.05, momentum 0.9 | 97.81% | 97.86% |
| AdamW + projection, alpha 1e-5 | 97.74%, 97.53%, 97.84% (seeds 1-3) | |
| AdamW + projection, alpha 1e-4 / 1e-3 | 97.17% / 92.52% (too strong) | |
| AdamW + projection, normalized, alpha 1e-3 / 1e-2 | 97.79% / 97.74% | 97.83% / - |

**The C reference** (`bench/ref/mlp.c`): the same network in C over OpenBLAS - same random numbers (xoshiro256**
seeded by splitmix64), same initialization order, shuffle, scaling and AdamW - so its losses match oann's to four
decimals in the first epoch, and the time is the comparison. PyTorch could not be installed: download.pytorch.org is
refused by the container's network policy (403).

**An epoch** (469 steps), seconds, medians of interleaved rounds (`make epoch`), on the shared 4-core machine - with
olang ef939ae at a load average of 4-5, with 9621af3 at 10-12 (other agents compiling: the 4-thread figures lose to
the oversubscription, OpenBLAS's worst):

| | 1 thread, ef939ae | 4 threads, ef939ae | 1 thread, 9621af3 | 4 threads, 9621af3 |
|---|---|---|---|---|
| oann, built for this machine (olang's default since 9621af3) | - | - | **0.69** | 1.57 |
| oann, built for baseline x86-64 (the default before; `-a x86-64` since) | 2.39 | 2.90 | 2.24 | - |
| oann, ef939ae's IR relinked with `-march=native` | 1.92 | - | - | - |
| C over OpenBLAS | 0.79 | 1.19 | 0.93 | 4.53 |

What changed is olang: 9621af3 builds for the machine it runs on (B12) and std/linalg tiles its GEMM for it (12 x 32
F32 accumulators in AVX-512 registers, multiply-adds fused where the target has FMA), and oann's products now go
through std/linalg's `GemmWorkspace` instead of a copy of its old 4 x 12 SSE kernel - a copy that, built native, ran
3x slower (5.1-5.4 s an epoch: LLVM's SLP vectorizer grouped its accumulators across rows at 512 bits). At this
problem size - 128 x 784 x 128, 128 x 128 x 10 - the native GEMM runs past OpenBLAS's, single-threaded and
interleaved. Where the epoch goes now (`bench/train.olang`, best of three, at a load of 12): forward 47%, backward 35%,
the optimizer 10%, filling batches 8% - 0.60 s in all; with the products this much faster, filling and the optimizer
(3% and 4% before) are a sixth of the epoch. The projection
adds ~0.3 s an epoch (one GEMM per dense layer per step, into a workspace of its own). Results are identical for
every thread count (GEMM partitions its output).

**Memory.** A step allocates nothing: the arena is laid out once, and the products pack into the graph's
`GemmWorkspace`, which reaches its size in the first step. (Before olang 9621af3 std/linalg's `Gemm` packed into
scratch arrays of its own scope, and the runtime's chunk pool, reusing only its head chunk, mapped a new B panel for
every product - 8.1 MB a step for the transformer of section 11. oann carried a copy of the algorithm packing into its
arena until the runtime's pool was fixed (olang O8b, size classes) and std/linalg gained `GemmWorkspace`, both asked
for here.) The transformer's resident memory grows by 68-76 kB over its first 20 steps and by nothing after.

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
  to 1e-12); the schedule, clipping, checkpoints and the text corpus.
- `make lmref`: the first steps of the character model - losses and gradient norms - against the same model, the same
  starting values, windows, AdamW, schedule and clipping in numpy (`bench/ref/charlm.py`, F64).
- `make lmbench`: where a transformer's step goes, by kind of operation (the graph's own profiling).
- `make mnist` trains end to end (needs the downloaded data); `make epoch` times it against C.
- Phase 4 (section 16): `tokenizer.olang` (GPT-2's chunks, merge order and ties, round trips, JSON), `checkpoint.olang`
  (safetensors in every dtype, each error, GPT-2's names), `conv.olang` (the convolution against its definition, col2im
  as im2col's adjoint, max pooling) and `vision.olang` (convolution and pooling against central differences); `make
  bpe` and `make safetensors` check the tokenizer and the format against independent Python implementations.

## 10. Serialization

Checkpoints as **safetensors** (`checkpoint.olang`, section 16): an 8-byte header length, a JSON header naming each
matrix's dtype, shape and byte range, then the raw little-endian data - interoperable with PyTorch, numpy and Hugging
Face, written with the floats' `Bits()`. Parameters carry PyTorch's names (`0.weight`, `h.3.attn.q_proj.bias`) given by
a `checkpoint.Names`. `g.Save`/`g.Load` stay as the raw parameter region, for oann alone.

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
  spread over the graph's threads; each task has its own scratch in the arena and its own `GemmWorkspace`.
- **Attention's products are std/linalg's `Gemm`**, one per sequence and head and product (section 18): the forward's
  `S = Q K^T / sqrt(dh)` and `Y = P V`, the backward's `dP = dO V^T`, `dQ`, `dK` and `dV` - six products of `T x T x dh`
  on the heads' strided views, nothing copied but a transposed block (below). **Causal masking stays exact**: the whole
  square of scores is computed, then each row's softmax runs over positions `0 ... i` and writes zeros after, and
  `dS` is zero there too, so a later row enters no product (zero times a finite value adds nothing) - a row's output
  and its gradients are bit for bit what they are without the later rows (checked). Half of every product is that
  upper triangle, wasted; a batched, causal-aware `Gemm` in std/linalg (section 13) would skip it, and `attendHead` and
  `attendHeadBackward` in `ops.olang` are all that would change. A product whose right operand is a head's `T x dh`
  block takes it transposed into the task's scratch first: std/linalg computes a product of at most 64^3
  multiply-adds directly when its right operand is not transposed, and at `T` 64 and `dh` 32 that ran 2.0-2.3x slower
  than the transpose and a packed product together (17-21 us against 8-10, a head's `P V`). Before this the products
  were dot products and updates over the heads' views (`kernels.Dot`, `kernels.Axpy`), computing only the causal half:
  faster than a `Gemm` per head on the baseline target, slower on this machine's AVX-512 (`bench/attention.olang`,
  section 18).
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
  count, then little-endian bits); safetensors when a model has to leave oann (section 16).

### Validation

Every new backward agrees with central differences in `F64` (section 9). Against a reference: `bench/ref/charlm.py`
is the character model again in numpy, F64, with a hand-written backward, started from the **same parameters** -
xoshiro256** and splitmix64 in Python, the same draws in the same order, rounded to F32 as oann stores them - and fed
the **same windows**, with the same AdamW, schedule and clipping. The first ten steps (`make lmref`), oann in F32
against numpy in F64:

| step | loss, oann | loss, numpy | gradient norm, oann | gradient norm, numpy |
|---|---|---|---|---|
| 1 | 4.211253 | 4.211253 | 7.050046 | 7.050046 |
| 2 | 4.181737 | 4.181737 | 6.326454 | 6.326455 |
| 5 | 4.033323 | 4.033323 | 5.406032 | 5.406031 |
| 10 | 3.751469 | 3.751469 | 1.995772 | 1.995772 |
| 20 | 3.552264 | 3.552264 | 1.526185 | 1.526185 |
| 40 | 3.034891 | 3.034891 | 1.233417 | 1.233417 |

Over 40 steps the losses differ by at most 1e-6 and the norms by 2e-6 - what F32 against F64 arithmetic allows - so
the forward, every backward, AdamW, the schedule and the clipping are the reference's. (Run again with olang 9621af3,
whose products fuse their multiply-adds: the same bounds, one norm in the table moving by its last digit.) One thing this found: the
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
second** - on the shared 4-core machine at a load average of 6 to 9 (other agents compiling and testing), with olang
ef939ae; built by 9621af3 for this machine a step is ~200 ms, 4,900-5,200 tokens a second on one thread (below). Still falling
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

`make lmbench ARGS="20 1"` - the graph's own profiling, ms a step, one thread: with olang ef939ae (baseline x86-64) at
a load average of 2.6, and with 9621af3 (this machine's AVX-512) at 13 - one of five runs, which gave 195-241 ms a
step:

| | forward, ef939ae | backward, ef939ae | forward, 9621af3 | backward, 9621af3 |
|---|---|---|---|---|
| `Linear` (12 products a step forward, 24 backward) | 138.5 | 281.5 | 37.9 | 81.7 |
| `MatMul` (the tied head) | 1.5 | 3.0 | 0.7 | 0.8 |
| `Attention` | 20.3 | 24.7 | 28.2 | 27.3 |
| `GeluTanh` | 12.5 | 14.9 | 5.3 | 6.9 |
| `LayerNorm` | 2.9 | 4.2 | 4.1 | 5.0 |
| `Add` (residuals) | 1.3 | 1.8 | 1.5 | 2.2 |
| the loss, embedding, positions | 0.8 | 0.3 | 1.2 | 0.5 |
| clipping, AdamW | 4.0 (together) | | 3.7 (together) | |

With ef939ae a step was **512 ms: products 83%, attention 9%, the rest of the graph 8%**, clipping and the optimizer
under 1%; the products ran at ~11 GFLOPS (4.8 GFLOP a step) - std/linalg's GEMM on the SSE2 target, slower than its
14-17 on large squares because these are narrow (k = 128). Two threads: 399 ms (2,566 tokens a second), four: 344 ms
(2,977) - the products scaled (424, 326, 275 ms), the element-wise ops did not (they run on one thread).

With 9621af3 a step is **~200 ms - 4,900-5,200 tokens a second on one thread, 2.5x the old step - products 58%,
attention 27%, the rest of the graph 13%**: the products run at ~40 GFLOPS (std/linalg's 12 x 32 AVX-512 tile with
FMA), and GELU's `FastTanh` pass, on the wider vectors, fell from 27 to 12 ms. Attention, a third of what is left, did not
move - its dot products were already vector loops - and a `Gemm` per sequence and head now beats them (section 11's
attention bullet, section 13). On two or four threads this load made the products 2.4x *slower* (277 and 273 ms of a
350 ms step): every product joins its tasks at each block, and with 13 runnable processes on 4 cores a task waits
for a core - the same oversubscription that took OpenBLAS's 4-thread MNIST epoch from 0.93 to 4.53 s (section 8);
thread scaling wants measuring on a quiet machine. The arena is 86.0 MB, laid out once (87.2 with the old packing
region in it); resident memory grows by 68-76 kB over the first 20 steps and by nothing after.

### olang issues found on the way (repro/)

- `repro/chunkpool.olang` (phase 2) - std/linalg's `Gemm` leaked a chunk per packed product through the runtime's
  pool; for a transformer 8.1 MB a step. Worked around by a copy of the algorithm (`kernels.Gemm`) until olang 9621af3
  fixed the pool (O8b) and gave std/linalg `GemmWorkspace`; the reproducer and the copy are gone.
- `repro/capturedfn.olang` - a lambda that captures a function value and calls it (`fn(d, g, v) { return d + f(g, v)
  }`) leaves an indirect call per element even when everything is inlined: 3.1 ns an element against 0.55 written
  out (still on 9621af3: 3.7-4.9 against 0.6-1.1 at a load of 8). `ops.backward2` runs its loops itself.
- `repro/ctorunstored.olang` - O26 counts an instance as referring to every reference its constructor was given, even
  one it only reads: `return Counts(t)` with `t` a local is refused (still on 9621af3). `text.Load` declares its text
  `&return` instead.
- Not issues, recorded for the language's records: `:=` from a comparison (`ta := t % 2 == 1`) needed its type
  written (D15 - relaxed since: `:=` takes any settled expression); a `match` value cannot give several results
  (`=> rows, cols`), so `savedShape` uses statements; a text join's piece cannot be a conditional (`$` of a local
  holding it is).

### What remains

- **Mixed precision** - BF16 storage with F32 master weights - not started: the graph is generic over one element type
  (section 14). (std/linalg's BF16 product packs into a `GemmWorkspace` now, widening to F32 as it packs.)
- Dropout of attention's probabilities (nanoGPT's `attn_dropout`): only the residual branches and the embedding are
  dropped out.
- A tiled attention that recomputes `P` in the backward instead of saving `T x T` per head (memory at long contexts),
  and a batched causal `Gemm` under it (section 13).
- Fused QKV and splitting heads with view nodes; a `Gemm` epilogue for bias and GELU (section 13).
- Rotary positions; generating several sequences at once (a decoding graph of `B` sequences). (The byte-pair
  tokenizer is built: section 16.)
- Reusing activation storage by liveness: the arena holds every activation and gradient - 87 MB for the model above.
- Element-wise ops over threads: at four threads they are 40 of the 344 ms.

## 12. Shapes as type parameters (when olang has const generics)

olang is getting `Matrix<T, R, C>`, each dimension a constant or known at run time. oann is ready for it: a layer's
shapes are given once, at construction, and `Apply` takes only handles, so `Dense<T, In, Out>` can hold
`W Matrix<T, Out, In>` and a handle carry its column count (`Var<C>`), making a shape mismatch in a model a
compile-time error. The executor keeps run-time shapes, as JAX keeps shapes as data while checking them when tracing.

## 13. What oann needs from std/linalg next

oann does all its arithmetic through `linalg.Matrix<T>` - `View`, `Row`, `GemmWorkspace` and its `Gemm`, `Gemv`,
`Map`/`Map2`/`Map3`, `AddRow`, `ColumnSums(out, beta)`, `RowSoftmax`, `RowNorms`, `ArgMaxRows`, `Fill`,
`FillUniform`/`FillNormal`, `Cast`, the `Fast*` functions. Done since this list was written (olang 9621af3): **a
`Gemm` into a caller's workspace** (`linalg.GemmWorkspace<T>`, which grows where it lives to the largest product
given - the graph keeps one, and `kernels.olang`'s copy of the algorithm is gone), **micro-kernels for the machine**
(per-target tiles, FMA where the target has it, the compiler building for the machine it runs on: an MNIST epoch
2.39 -> 0.69 s, the transformer's products ~11 -> ~40 GFLOPS) and **two-operand indexing** (`m[r, c]`, E31's
multi-index - not used by oann, whose element access is `Get`/`Set` and runs of rows). What it still needs:

1. **Batched strided `Gemm`, causal-aware** - every sequence and head of attention in one call, the products of a
   lower-triangular block skipped. Attention runs on a `Gemm` per sequence and head now (section 18): 29 of ~150 ms a
   step at context 64, 73 of ~180 at 256, growing as `T^2` - and half of each product is the upper triangle the
   softmax then throws away. A batched causal `Gemm` would skip it and make one call where there are six per sequence
   and head; `ops.attendHead` and `ops.attendHeadBackward` are where it goes in. Also: **std/linalg computes a product
   of at most 64^3 multiply-adds directly** (no packing) when its right operand is not transposed, and on this
   machine's AVX-512 that path is 3x slower than the packed one at attention's `64 x 32 x 64` (17-21 us against 6.5);
   attention transposes its right operands to reach the packed path. The threshold wants lowering on wide targets.
2. **`Gemm` with an epilogue** - `c = act(alpha op(a) op(b) + bias row)` - so `Linear` plus an activation is one pass:
   the bias row and GELU are two more passes over the widest activation (B T x 4D) - GELU's forward alone was 12.5 ms
   of a 512 ms step and its backward 14.9, ~5%; with the products 3.5x faster on the native target, 12 of ~200 ms; and
   **fused QKV** (one product of width 3D, attention reading q, k and v as column blocks) needs view nodes in oann's
   graph, not linalg.
3. **For convolutions** (section 16): a `Gemm` whose A panels are packed straight from the images (an implicit GEMM,
   or a packing callback) - im2col and the product's packing copy every patch twice, and the patches matrix is the
   largest convolution's whole im2col; and a path for thin products (a depth of 9 for a one-channel first layer runs
   at half the rate of the others).

## 14. Open questions

1. The projection: normalized, alpha 1e-3 by default (97.79% on MNIST, section 8); still open whether alpha should
   scale with the learning rate and whether it should apply to every dense layer or only some.
2. An eager mode beside the graph, for models whose structure depends on their data - not before a model needs one.
3. oann as an importable package (`github.com/OWNER/oann/nn`) - the layout already is one module per concern.
4. Mixed precision (BF16 storage, F32 master weights): the graph is generic over one element type, so it needs either
   a second arena of BF16 copies the products read (std/linalg's BF16 product packs into a `GemmWorkspace` too,
   widening to F32 as it packs) or a graph with a type per node. Not started (section 11).
5. View nodes - a node whose value is a column block of another's, no storage of its own - would give fused QKV (one
   product of width 3D) and splitting heads at no cost; the "written" flag per node would then need to be per block.

## 15. Layout

```
makefile                OLANG ?= the compiler; make test, data, mnist, cnn, epoch, charlm, lmbench, lmref, bpe,
                        bpelm, safetensors, clean
nn.olang                Graph, Var, Op, Node: recording, Plan, Forward, Backward, decoding, checkpoints,
                        profiling; the gradient checks
ops.olang               the kernels: forward and backward per primitive, on Matrix
kernels.olang           Dot, Axpy, Sum over runs of arrays (decoding's attention, the normalizations), SquaresF64
layers.olang            Dense, Mlp, activations, PyTorch's initialization; Norm, AttentionBlock, Transformer
optim.olang             the Optimizer trait, Regularizer, AdamW (and AdamWProjected), Sgd; WarmupCosine, ClipGradNorm
train.olang             Classifier, Epoch, Evaluate; LanguageModel, Gpt, LmStep, LmEvaluate
generate.olang          sampling; generation running the context again, and with a key-value cache
tokenizer.olang         byte-level BPE: GPT-2's pre-tokenization, training, encoding, decoding, JSON
checkpoint.olang        safetensors: Names (PyTorch's for the layers), Save and Load in F64, F32, F16 or BF16
conv.olang              convolution (im2col, one product, col2im) and max pooling on images held as rows
vision.olang            the convolution layer; the gradient checks of convolution and pooling
datasets/idx.olang      the IDX format
datasets/loader.olang   Labeled sets and the Loader
datasets/mnist.olang    fetching and loading MNIST
datasets/text.olang     a character corpus, its split, and windows of tokens; fetching tiny Shakespeare
examples/mnist_mlp.olang
examples/mnist_cnn.olang  a small convolutional network on MNIST, its memory and where its time goes
examples/charlm.olang   the character-level transformer on tiny Shakespeare, trained, saved and sampled
examples/charlm_sample.olang  sampling from its checkpoint, with and without the cache
examples/bpelm.olang    the same transformer on 512 BPE tokens, per character against the character model
bench/data.olang        the data pipeline, checked and timed
bench/train.olang       where an epoch's time goes
bench/epoch.sh, ref/    an epoch against the C reference over OpenBLAS
bench/lm.olang          where a transformer's step goes, by kind of operation
bench/lmref.olang, ref/charlm.py  the first steps of the character model against numpy
bench/attention.olang   attention's Gemm per sequence and head against the dot-product loops it replaced
bench/abstep.py         two builds of bench/lm.olang run in alternation: the medians of their steps
bench/bpe.olang, ref/bpe.py  BPE timed, and checked against an independent Python implementation
bench/conv.olang        a convolution's passes timed: im2col, the three products, col2im
bench/safetensors.olang, ref/safetensors_check.py  safetensors both ways against numpy
docs/settling.md        settling networks - the model after transformers
repro/                  minimal programs for olang issues found here
data/, build/           downloads and build output, not in git
```

Each file is one olang module, imported by its path relative to the importing file without the extension
(`import "../datasets/mnist"`). Random numbers are std/rand's, times std/time's.

## 16. Phase 4: byte-pair tokens, safetensors, convolutions

### Byte-level BPE (`tokenizer.olang`)

GPT-2's tokenizer - Sennrich et al.'s byte-pair encoding, run on bytes as Radford et al. run it: the 256 byte values
are the first tokens and every later token is the merge of two earlier ones, so any bytes encode and decode back to
themselves and there is no unknown token.

- **Pre-tokenization** - the chunks merges never cross - is GPT-2's pattern,
  `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, read by a hand-written scanner
  (`ChunkEnd`; olang has no regular expressions) with byte classes standing for the Unicode ones: a letter is an ASCII
  letter or any byte from 0x80 up (so a UTF-8 word stays one chunk), a digit is 0-9, whitespace the six ASCII spaces.
  On ASCII text - tiny Shakespeare is ASCII - this is GPT-2's rule exactly; elsewhere it keeps a non-ASCII symbol or
  punctuation mark with the letters around it, where GPT-2 would split it off.
- **Training** (`tokenizer.Train(text, vocab)`) counts each distinct chunk once with its number of occurrences, then
  merges the most frequent adjacent pair again and again, ties to the smaller pair (Hugging Face's rule). Pair counts
  are kept up to date as chunks change, a heap with lazy deletion gives the most frequent, and every pair keeps a list
  of the chunks it occurs in (a linked list in flat arrays), so a merge costs the chunks it touches, not the corpus.
- **Encoding** (`t.Encode(text)`) applies a chunk's merges lowest rank first, leftmost first, with a heap over its
  positions: the tokens of GPT-2's "merge every occurrence of the best pair, left to right" (a merge only ever makes
  pairs of a later rank), in O(n log n) of the chunk's length - a megabyte of one letter is no harder than prose, where
  the usual quadratic loop would not finish. Each distinct chunk's tokens are remembered for the rest of the text, as
  GPT-2's encoder does. The ranks are an open-addressed table of pairs. `t.Decode(ids)` fails on an id the vocabulary
  does not have (`TokenizerError.UNKNOWN`).
- **Saving** is JSON, `t.Save(path)` and `tokenizer.Load(path)`: `{"format": "oann-bpe", "version": 1, "pretokenize":
  "gpt2-ascii", "merges": [[32, 116], ...], "vocab": [...]}`. The merges, as pairs of ids, are what Load reads (each
  checked to name only tokens made before it); the vocabulary is every token in GPT-2's printable form ("Ġthe" for
  " the"), for a person to read.

**Checked** by the module's tests (GPT-2's chunks on contractions, runs and spaces; the merge order and its ties; a
round trip of random bytes of all 256 values, of random corpus text and of 100,000 of one letter; saving and loading)
and **independently** by `bench/ref/bpe.py` (`make bpe`): the same pre-tokenization as a Python bytes regular
expression, the plain algorithm (every pair counted again for every merge) and GPT-2's encoder. From tiny
Shakespeare's training part both learn **the same merges** - 256 of 256 at a vocabulary of 512, 1,792 of 1,792 at
2,048 - and give **the same validation tokens** (59,401 and 43,559).

**Speed**, tiny Shakespeare's first 90% (1,003,854 bytes: 266,995 chunks, 14,134 distinct), one thread on the shared
machine (load average 3-6, so ranges):

| vocabulary | training | characters a token (training / validation part) | encoding the training part |
|---|---|---|---|
| 512 | 50-95 ms | 1.944 / 1.878 | 35-60 ms (17-28 MB/s) |
| 1,024 | 131 ms | 2.442 / 2.257 | 63 ms |
| 4,096 | 120 ms | 3.264 / 2.903 | 43 ms |
| 8,192 | 152 ms | 3.535 / 3.186 | 50 ms |

A merge costs only the chunks it touches, so 7,936 merges cost about what 256 do: training is mostly counting the
chunks (10-15 ms) and building the pair lists. The Python reference takes 12 s at 512 and 80 s at 2,048.

### A transformer on BPE tokens

`examples/bpelm.olang` (`make bpelm ARGS="2000 1 2"`) is the character model of section 11 unchanged - 4 layers of
width 128, 4 heads, a context of 64, batches of 16 sequences, tied embeddings, AdamW (beta2 0.99, decay 0.1) warmed up
over 100 steps and down a cosine to 1e-4, clipping to 1, 2,000 steps, seed 1 - on 512 BPE tokens learned from the
training part (867,072 parameters: the embedding is 57,216 larger). The split is the character model's, so both see the
same text on each side. A loss per token becomes a loss per character as the total over the predicted tokens divided by
the characters those tokens hold, counted on the very windows evaluated (nats a token / characters a token).

| step | BPE, nats a token | BPE, nats a character | characters, nats a character |
|---|---|---|---|
| 0 | 6.2683 (ln 512 = 6.238) | 3.3206 | 4.2086 |
| 250 | 3.8604 | 2.0450 | 2.3825 |
| 500 | 3.6399 | 1.9282 | 2.2186 |
| 750 | 3.4664 | 1.8363 | 2.0828 |
| 1000 | 3.3325 | 1.7654 | 2.0032 |
| 1250 | 3.1944 | 1.6922 | 1.9575 |
| 1500 | 3.1254 | 1.6557 | 1.8779 |
| 1750 | 3.0825 | 1.6329 | 1.8220 |
| 2000 | 3.0242 | **1.6020** | **1.8063** |

Each row is the first 10 batches of the model's own ordered pass, so the two columns cover different text (10,240 BPE
tokens hold ~19,000 characters). Over the **whole validation part** (111,540 characters, the same text for both): the
BPE model **3.1095 nats a token, 1.6559 a character**, the character model's checkpoint **1.8404 a character** - 10%
lower for the same model, steps and batch shape. Some of that is context: 64 tokens are ~120 characters. 959 s on two
threads at a load average of 4 (480 ms a step, 2,136 tokens - ~4,150 characters - a second); the wider head (512 logits
against 65) costs little next to the blocks.

A sample (`generate.CachedTokens` - the key-value cache on token ids, for any tokenizer; `generate.Cached` is it with a
character corpus's encoding - 250 tokens, 512 characters, after a newline, temperature 0.8, 2,156 tokens a second):

```
I'll not him with your greations,
But now you cannot be bound under against their after
As or morrow; and I should else thee
The is wours.

DUKE VINCENTIO:
Tow,, traitor, to must be clove, which, and a done:
For she is the gentlemen?

CAMILLO:
I cannot I'll take it be her:
Nay, which heaven of my banish, drowns moes was
```

Found on the way: `text.Windows.At` past the end of an ordered pass gave a negative count, which `LmEvaluate`'s
`n == 0` test let through as a `Forward` of negative rows - it gives 0 now (evaluating a whole part asks past the end).

### safetensors (`checkpoint.olang`)

Hugging Face's format for tensors, so a model can leave oann for PyTorch, numpy or JAX, or come in from them: eight
bytes holding the header's length, little-endian; the header, JSON naming every tensor's `dtype`, `shape` and
`data_offsets` (its byte range of what follows), padded with spaces to a multiple of eight; then the tensors' elements,
row by row, little-endian.

```olang
names := checkpoint.Names()
names.AddTransformer(t, "")                       # GPT-2's names; AddDense, AddNorm, AddMlp for the other layers
try checkpoint.Save(g, "model.safetensors", names, checkpoint.Dtype.BF16)
try checkpoint.Load(g, "model.safetensors", names)
```

- **Names** are given to parameters, not kept by the graph: a `Names` maps parameter handles to text, and the layer
  helpers use PyTorch's: a dense layer's `weight` and `bias`, a perceptron as `nn.Sequential` numbers it (`0`, `2`,
  ...: an activation between every two), a transformer GPT-2's (`wte`, `wpe`, `h.I.ln_1`, `h.I.attn.q_proj`, ...,
  `h.I.mlp.c_fc`, `h.I.mlp.c_proj`, `ln_f`, `lm_head` when untied - q, k and v are three products here, so they are
  named as Llama's are). A parameter given no name is `param.K`, K its place among the parameters. Keeping names out of
  the graph leaves `nn.olang` and its nodes plain data; two parameters given one name are an error (`DUPLICATE`).
- **Layout** is PyTorch's: weights `[out, in]` (nn.Linear's; note GPT-2's own Conv1D checkpoints are `[in, out]`), and
  a parameter of one row - a bias, a gain - is written as a vector, `[cols]`, as PyTorch holds it; either form reads
  back. A tied head is the embedding itself, so it is one tensor, as Hugging Face writes tied weights. The metadata is
  `{"format": "pt"}`, which Hugging Face's loaders look for.
- **Types**: `Dtype.F64`, `F32`, `F16`, `BF16`, or `Same`, the graph's own (the default). Writing rounds once - every
  element is exact as an F64, then rounded to the narrower type, nearest and ties to even - and reading converts any of
  the four into the graph's type the same way.
- **Loading is strict**, as PyTorch's `load_state_dict` is by default: every parameter must be in the file with its
  shape and size (`MISSING`, `SHAPE`), the file may hold nothing else (`UNEXPECTED`), a header that is not JSON of
  tensors or a range outside the data is `MALFORMED`, and another dtype is `DTYPE`.

**Checked** by the module's tests - a perceptron saved in every dtype and loaded into another graph (bit-exact in its
own type, equal to the rounded values in F16 and BF16), each error, a transformer's names and its tied head - and
**independently** by `make safetensors`: `bench/ref/safetensors_check.py`, a reader and writer of its own on `struct`,
`json` and `numpy` alone (no safetensors package - pip is blocked here, and the point is a second implementation), reads
oann's files in all four dtypes, checks the header (its length, the space padding, data ranges covering the data with no
hole or overlap) and compares **every element bit by bit** with the values rounded by numpy (F32, F16) and by
round-to-nearest-even on the F32 bits (BF16, which numpy lacks) - signed zeros, an F32 subnormal, F16's largest value
and the first past it (infinity), an F16 underflow, near F32's largest, BF16 ties - 144 elements. Then it writes a file
of its own - a tensor per dtype, in an order of its own, F16 subnormals and an F64 subnormal, a one-row parameter as a
vector, metadata of its own, an unpadded header - which oann loads into an F64 graph exactly: 27 of 27 values as numpy
stored them.

### Convolutions (`conv.olang`, `vision.olang`)

- **Images are rows, channels last.** An image is one row of `H x W x C` - a pixel's channels together, NHWC for the
  batch (TensorFlow's default layout) - and a convolution's output rows are `OutH x OutW x OutC` the same way. The
  weights are PyTorch's `Conv2d.weight` flattened, `OutC x (C K K)`, each kernel `[C][K][K]`, so a checkpoint maps
  onto PyTorch's by a reshape (a PyTorch model's flatten before its first dense layer is channels first, though: such
  a layer's columns need permuting).
- **One product for the whole batch.** im2col writes every output position of every image as a row of the patch it
  sees - `(n OutH OutW) x (C K K)`, zeros where the kernel hangs over the padding - and `Y = cols W^T + b` is one
  product (std/linalg's, into the graph's workspace) and a bias row. With channels last, `Y`'s `OutC`-wide rows are the images' output rows one after
  another: the node's value *is* `Y`, reshaped, nothing transposed. Channels first would need a product per image, or
  a transpose pass - which is why section 1's `C x H x W` plan changed.
- **Backward**: `db` = the column sums of `dY`, `dW = dY^T cols`, `dcols = dY W` (only when the input wants a
  gradient - a network's first convolution skips it), then col2im adds every patch's gradient back to the pixels it
  came from. The patches are **not kept**: the backward builds them again into the same region before `dW` (a pass
  over memory, against products of the layer's size) and `dcols` then overwrites them, so a graph has **one patches
  region, its largest convolution's**, shared by all of them; the graph's `GemmWorkspace` grows to their products in
  the first step. For the MNIST network below, 128 x 784 x 9 F32 - 3.6 MB.
- **im2col's innermost loop** is a pixel's channels, read in order - unless an image has fewer channels than the
  kernel is wide (a first layer, grey or RGB), when a window inside the image is copied a kernel row at a time,
  written in order with no test per element: 3.0 ms against 8.9-11.0 for a batch of 128 one-channel images, and the
  same as before for 16 channels (measured interleaved in one process).
- **`MaxPool2d(x, C, H, W, k, stride)`** keeps, as its saved matrix, where each maximum was in its window (`ky K +
  kx`, exact in any float type; the first of equal values, as PyTorch picks); the backward adds each gradient there,
  so overlapping windows add up. No padding yet. A pixel's channels are the innermost loop here too.
- Images are split over the graph's threads (`linalg.ParallelRows`), the products by std/linalg's `Gemm`. Builders:
  `g.Conv2d(x, w, b, inC, outC, kernel, stride, padding, H, W)`, `g.MaxPool2d(x, C, H, W, kernel, stride)`; the layer
  `vision.NewConv(g, inC, outC, kernel, H, W, stride, padding)` draws PyTorch's starting values (uniform in
  `+-1/sqrt(C K K)`, weights and bias).

**Checked**: the convolution through im2col and one product against its definition (four shapes - padding, strides 2
and 3, 1 to 3 channels, 2x2 to 5x5 kernels - on one task and three), col2im as im2col's adjoint (`<im2col(x), c> =
<x, col2im(c)>`), max pooling against its definition; and through the graph in F64 against central differences: two
convolutions reading one pooled value, a convolution applied twice (shared weights), overlapping 3x3 pools at stride
2, padding with stride 2, a short batch, and the input gradients of 3- and 1-channel images that are parameters. A
deliberately broken col2im fails both graph checks.

**MNIST** (`make cnn ARGS="10 1 1"`, `examples/mnist_cnn.olang`): a 3x3 convolution 1->16 (padding 1), ReLU, 2x2 max
pooling, a 3x3 convolution 16->32, ReLU, 2x2 max pooling, a dense layer 1568->10 - 20,490 parameters - with softmax
cross-entropy, AdamW at 1e-3 and batches of 128, seed 1, one thread - with olang ef939ae at a load average of 3-6, and
again with 9621af3 (products for this machine, multiply-adds fused) at 10-21:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| test accuracy, %, ef939ae | 96.97 | 97.83 | 98.42 | 98.55 | 98.78 | 98.41 | 98.52 | 98.81 | 98.83 | **98.84** |
| test loss, ef939ae | 0.0994 | 0.0652 | 0.0474 | 0.0442 | 0.0385 | 0.0457 | 0.0416 | 0.0351 | 0.0349 | 0.0334 |
| test accuracy, %, 9621af3 | 96.97 | 97.78 | 98.44 | 98.54 | 98.78 | 98.42 | 98.54 | 98.87 | 98.81 | **98.82** |
| test loss, 9621af3 | 0.0994 | 0.0650 | 0.0471 | 0.0444 | 0.0383 | 0.0451 | 0.0412 | 0.0351 | 0.0347 | 0.0334 |

against the 784-128-10 perceptron's **97.73%** after its 10 epochs (101,770 parameters, section 8). An epoch takes
**54-61 s** with ef939ae (57.6 s on average over the ten; 54.1 s after max pooling's channel loop moved inside), and
**37-53 s with 9621af3** (42.9 on average, at a load of 10-21 - other agents compiling; the epochs at the lower load
took 37-41 s), against the perceptron's 0.69 s: the network does 0.76 GFLOP a step to the perceptron's 0.05. Results are identical bit for bit on
one and two threads (two were slower on the oversubscribed machine). **A step allocates nothing**: resident memory
116,748 kB after step 10, 116,756 after step 100, 116,760 after every epoch to the tenth (4,690 steps); the arena is
63 MB, laid out once.

Where the first epoch goes (`ARGS="1 1 1 16 32 profile"`, ef939ae): the convolutions 46.0 s (forward 17.5, backward
28.5), max pooling 4.6, ReLU 2.4, the dense layer 1.0. Split by pass (`bench/conv.olang`, ms for a batch of 128; for
9621af3 the median of three runs at a load of 10):

| | im2col (twice) | forward product | dW | dcols | col2im | bias and its sums |
|---|---|---|---|---|---|---|
| conv 1->16 on 28x28, ef939ae | 6.3 | 5.4 (5.4 GFLOPS) | 4.7 (6.1) | 4.6 (6.3) | 3.0 | 1.7 |
| conv 16->32 on 14x14, ef939ae | 8.8 | 21.6 (10.7) | 19.6 (11.8) | 18.5 (12.5) | 4.1 | 0.9 |
| conv 1->16 on 28x28, 9621af3 | 9.5 | 5.4 (5.3) | 8.7 (3.3) | 4.7 (6.1) | 4.3 | 2.4 |
| conv 16->32 on 14x14, 9621af3 | 10.1 | 8.5 (27.1) | 11.5 (20.2) | 6.2 (37.6) | 4.7 | 0.7 |

(In the network the first convolution needs no `dcols` or col2im - its input is the images.) With ef939ae the second
convolution's products ran at 11-12 GFLOPS, std/linalg's GEMM rate on the SSE2 target; with 9621af3 they run at 20-38
at a load of 10, and the epoch is now spread over the passes rather than the products: im2col, col2im and max
pooling are passes over memory the target does not change. The first convolution's products stay at 3-6 GFLOPS: a
depth of 9 (`C K K` for one channel) is too thin for a packed product, and its `dW` (16 x 9, depth 100,352) is mostly
padding in the native 12 x 32 tile, slower than the old 4 x 12 one - the case for a direct convolution, or a
small-depth path in `Gemm`. And im2col plus the product's own packing copies every patch twice: a `Gemm` that packs
its A panels straight from the images (an implicit GEMM) would skip the patches matrix and both passes over it - the
larger lever now that the products are fast.

### olang issues found in phase 4 (repro/)

No compiler bug: everything phase 4 needed compiled as the spec says. Two frictions, both spec-conforming:

- `repro/nestedtry.olang` - a catch clause covers its own `try` only, so `try (try entry["shape"]).AsArr() catch {
  ... }` - look a member up, then convert it - lets the lookup's error escape. `checkpoint.member` converts it first.
- `repro/tryindex.olang` - `try doc[names[k]]` checks `names[k]` too (E16d), adding `BuiltinError` to what the try can
  fail with, though only the lookup was meant. Naming the element first avoids it.

And two limits worth a library: no regular expressions, so GPT-2's pre-tokenization pattern is a hand-written scanner;
and no Unicode character classes in std, so `\p{L}` and `\p{N}` became byte classes (exact on ASCII text). Also met
again: `:=` from a comparison needed its type written (D15 - relaxed in olang 9621af3).


## 17. Settling networks (`circuit.olang`)

Beside the graph, `circuit.Circuit<T>` is the second recorded structure: a network of regions whose neurons settle to
an equilibrium, taught by equilibrium propagation (contrasting a free settle with nudged ones) - same `Matrix`, same
one-arena discipline, nothing allocated per settle, lesson or step. Its neuron model is a closed enum with rate
neurons and a working leaky integrate-and-fire variant. Phase 1 is built: the circuit, the certificate and refusal,
`Teach`, gradient checks in F64 (the estimate's error falls as beta^2), XOR with rate and with spiking neurons, and
MNIST at 97.5% with a 784-128-10 circuit. The design, the decisions taken building it and the measurements are
docs/settling.md (sections 2.10, 4.6, 5.7, 6). `make test` runs its checks; `examples/xor_settle.olang`,
`examples/xor_spiking.olang` and `examples/mnist_settle.olang` run it end to end.

Phase 2 is built in `agent.olang`. `agent.Agent<T>` is a circuit living one moment at a time, with these parts:

- a three-factor actor: the circuit, an eligibility trace and a TD error;
- a linear critic;
- a context trace;
- a fast and a persistent associative memory;
- an arousal gate. Calm moments act greedily and touch no synapse; aroused moments explore and learn.

An agent's whole state checkpoints to a safetensors file and restores exactly. `optim.FlatAdamW`/`FlatSgd` step a
circuit's flat parameter region with the same loops `AdamW`/`Sgd` use.

`examples/bandit_settle.olang` runs the contextual bandit, reversal and trace-pinning tasks over many lives with
confidence intervals. The results and decisions are in docs/settling.md, sections 2.11, 5.8 and 6.

## 18. Phase 5: attention on products, mixed precision

### Attention through std/linalg's Gemm

Attention's six products per sequence and head - `S = Q K^T`, `Y = P V`, `dP = dO V^T`, `dQ = dS K`, `dK = dS^T Q`,
`dV = P^T dO` - are each one `ws.Gemm` on the heads' strided views (section 11), replacing the dot products and
updates over the same views, which computed only the causal half. A task packs into a `GemmWorkspace` of its own
(the graph keeps one per thread, grown at `Plan` by running the products once on zeros), and transposes a right
operand into its scratch first where std/linalg would otherwise compute the product unpacked (section 13). The softmax
stays `math.Exp` (a library call per element): `linalg.FastExp` in that loop measured slower, 3.6-3.8 ms against
2.9-3.1 for the forward at T 64, the running sum keeping it from vectorizing.

**Checked**: the gradient checks of section 9 (causal and not, two and four heads, a node attending to itself, short
batches), and two more through the packed products - sequences of 8 (the scores packed, the rest direct) and 130
(every product packed) against central differences in `F64` with a step of `1e-5` (at `1e-6` the differences' own
rounding is already 4e-7 to 7e-7 of these gradients), and causality through the packed products: changing row 25 of a
sequence of 32 leaves rows 0-19 of the output exactly as they were. The first 40 steps of the character model
against numpy (`make lmref ARGS=40`): losses within 1e-6 and gradient norms within 2e-6, as before.

**Measured** (olang edf8238, one thread, on the shared machine at a load average of 5-8): `bench/attention.olang`,
16 sequences of 64, width 128, 4 heads, medians of 9 alternating rounds - the forward 4.6-6.0 ms by the loops against
2.5-3.1 by `Gemm`s, the backward 4.5-5.2 against 3.1-3.7; at T 256 (16 sequences), 70-81 ms against 32-42 forward
and 71-83 against 36-37 backward. Without the transposes the products below 64^3 ran unpacked and the backward at T 64
was *slower* than the loops (6.0-7.5 ms against 4.5-5.1). The transformer step (`bench/lm.olang`, 10 steps after 3,
medians of 8 runs alternating the build before and after, `bench/abstep.py`):

| context | attention before | attention after | step before | step after |
|---|---|---|---|---|
| 64 (16 sequences) | 44.1 ms (21.6 forward, 22.5 backward) | 29.1 (13.4, 15.7) | 164.6 ms | 153.1 |
| 256 (4 sequences) | 150.6 (76.2, 74.4) | 73.4 (35.7, 37.7) | 264.3 | 182.1 |

(At 256, medians of 6 runs of 5 steps.) Attention is 1.5x faster at context 64 and 2.05x at 256; the step 1.08x and
1.45x. The products and the rest of the graph did not move (96-100 and 22 ms at 64).

### olang issues found in phase 5 (repro/)

- `repro/spawncall.olang` - a task spawned as an immediately called lambda, `spawn fn() { ... }()` in a loop, that
  captures two references and the loop's variable sees wrong captures: hundreds to thousands of 25,600 elements
  wrong. `spawn fn() { ... }` is refused for capturing the loop's variable (P2), so the called form seems to slip past
  that check. `ops.parallel`'s form - the lambda made once, outside the join - is right, and is what attention uses.

