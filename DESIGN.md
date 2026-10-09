# oann - design

oann is a neural-network library written in olang: layers, automatic differentiation, losses, optimizers, datasets and
the training loop, on top of the standard library's 2-D `Matrix<T>` (std/linalg or std/matrix). The C version this
replaces (dense layers, ReLU, softmax cross-entropy, AdamW, MNIST over OpenBLAS) is a rough reference for scope only:
it does not compile as it stands, and its API is not carried over.

The yardstick is olang's own (PRINCIPLES.md in the olang repository): no manual memory management, C-like performance
with no cost the code does not show, natural language, minimal syntax. Concretely for oann: **a training step
allocates nothing**, every hot loop is a GEMM or one fused pass over memory, and the user writes a model the way
PyTorch or Flax users would expect to.

## 1. The operand: Matrix

Every value oann computes with is a `Matrix<T>`: **rows are samples (the batch), columns are features**. There is no
N-d tensor - data is interpreted in one place only, by the operation that consumes it.

- **A sample is a row.** An MLP layer maps `[B, in]` to `[B, out]`; a loss reads one row per sample.
- **Weights are `[out, in]`** (PyTorch's layout, so checkpoints map one to one onto PyTorch's and Hugging Face's):
  a dense layer is `y = x W^T + b`, its backward `dx = dy W`, `dW = dy^T x`, `db` = the column sums of `dy`. All three
  are one GEMM with transposes as parameters (BLAS style: `NT`, `NN`, `TN`) - nothing is ever transposed in memory.
- **A bias is a `1 x C` matrix**, added to every row by an explicit row-broadcast operation; there is no implicit
  broadcasting.
- **Views are by row stride.** A range of rows (`m.Rows(0, n)`) and a block of columns (a stride larger than the
  width) are views of the same storage, made without allocating.
- **Structure beyond two dimensions lives in the operation's parameters.** A convolution is told its input's
  `C, H, W` and reads each row as one image; it lowers to im2col into a planned workspace matrix and one GEMM. Attention
  works on `[B*T, D]` matrices: each head is a column block (a strided view), and the per-head products are GEMMs on
  those views. Neither needs a tensor type.

oann is generic over the element type `T` from the start (`Graph<T>`, `Dense<T>`): MNIST trains in `F32`, gradient
checks run in `F64`, and `BF16`/`F16` storage with `F32` accumulation is the mixed-precision path later.

## 2. Automatic differentiation

### Decision: a recorded graph, replayed every step; each primitive has a hand-written backward

The model is run **once**, at build time, against a `Graph`: every operation appends a node (an op and its input
handles) and gets a handle back. `Plan()` then infers and checks every shape, decides which nodes need gradients,
and lays every value, gradient and workspace buffer out in **one arena**, allocated once. A training step is
`Forward(n)` (the nodes in recorded order) then `Backward(loss)` (in reverse, each op's vector-Jacobian product), then
the optimizer's step - none of which allocates.

Each primitive op has a hand-written backward (its VJP), as in every autograd system; the graph is what composes them,
so a new model, a residual connection, parameter sharing or a custom loss needs no new backward code.

### The alternatives, and why not

1. **Layer-level hand-written backward** (the C version, Caffe, Darknet): each layer type implements forward and
   backward over its whole computation and a sequential container calls them. Simple and fast, but every new layer
   needs its own backward written by hand, anything that is not a chain (a residual, a shared embedding, two losses)
   needs a special layer, and losses and optimizers are special-cased. It does not compose, and olang's lack of
   run-time interfaces would make the container an enum of every layer type there is.
2. **An eager tape** (PyTorch's define-by-run): each op runs immediately and records itself with the tensors its
   backward needs. Maximally flexible - Python control flow is the model - but every step allocates every activation
   and tape node anew. olang's arenas would make that cheap (one block scope per step, reclaimed at its end), but not
   free, and the per-step structure rules out planning: no buffer reuse, no fusion, no thread decisions made once.
3. **A recorded static graph** (JAX's jit, XLA, TensorFlow graphs, TFLite's arena planner, tinygrad's schedule) - the
   choice. Shapes are fixed per graph, which is what MLPs, CNNs and transformers training at a fixed batch size have;
   a shape change (a new batch size, a sequence length) re-records, as a jit retraces.

### Why it fits olang

- **Handles, not references.** A node is plain data: `Var` is a declared type over `I64` (nominal, so a handle is
  never mixed up with a count, and it indexes the node table directly). The graph owns a `List<Node>` of numbers and
  enums and one arena `Array<T>&`; nothing in the graph holds a reference, so olang's scope checker has nothing to
  prove and the graph is trivially serializable - the same design as XLA's instruction ids or tinygrad's UOp lists.
- **The op set is a closed enum, dispatched by `match`.** `Op` has one case per primitive with its inputs as
  payload (`MatMul(a Var, b Var, transA Bool, transB Bool)`, `AddRow(x Var, bias Var)`, `Relu(x Var)`, ...). olang
  compiles a `match` to a switch, which the olang benchmarks measured at C's speed - no virtual dispatch anywhere.
  User-defined ops are one `Custom` case holding forward and backward function values.
- **Views are values.** `g.Value(v)` and `g.Grad(v)` give a `Matrix<T>` view of the arena by value - no allocation,
  checked by a prototype that ran two million forward passes in constant memory (1.9MB resident).
- **No inheritance needed.** Layers are ordinary structs holding parameter handles with an `Apply(g, x) Var` method;
  composition is ordinary code. Traits are used only as constraints (an optimizer passed to a training loop is a
  `<O Optimizer>`), so everything is statically dispatched and inlined.

### Graph semantics

| oann | meaning | PyTorch analogue |
|---|---|---|
| `g := nn.Graph<F32>(128)` | a graph planned for batches of up to 128 rows | - |
| `x := g.Input(784)` | a `[B, 784]` input the caller fills each step | a batch tensor |
| `y := g.Classes()` | `B` class indices (`I32`) | `targets` |
| `w := g.Param(out, in, init)` | a trainable `[out, in]` matrix | `nn.Parameter` |
| `g.MatMul(a, b, transA, transB)`, `g.AddRow(x, b)`, `g.Relu(x)`, ... | record an op, give its handle | ops in `forward` |
| `g.Plan()` | check shapes, lay out the arena, allocate it - once | `torch.compile` |
| `g.Forward(n)` | run the batch's first `n` rows | `model(x)` |
| `g.Backward(loss)` | parameter gradients of `loss`, written fresh | `zero_grad(); loss.backward()` |
| `g.Backward(loss, accumulate)` | added to the gradients already there | `loss.backward()` without zeroing |
| `g.Value(v)`, `g.Grad(v)` | views of a node's value and gradient | `.data`, `.grad` |

- **A shorter batch** (the last of an epoch) runs on the first `n` rows of every batch-shaped buffer - views, no
  reallocation; losses average over `n`.
- **Gradients**: a node used by several ops gets the first contribution written and the rest added (decided at
  `Plan` from use counts), so no zero-fill pass is needed; parameters' gradients are written fresh each `Backward`
  unless accumulation is asked for (gradient accumulation over micro-batches).
- **What needs a gradient** is decided at `Plan`: a node needs one when it depends on a parameter; inputs and labels
  never do, so the first layer computes no `dx`.
- **Training and evaluation**: `g.Training` switches dropout (and later batch-norm statistics); evaluation is
  `Forward` alone.

### The arena

One `Array<T>` per graph, in five regions, each laid out at `Plan`:

1. **parameters**, contiguous in registration order - the optimizer's single flat view, and the checkpoint;
2. **parameter gradients**, the same layout - one flat view for the fused optimizer step and for gradient clipping;
3. **activations** (every batch-shaped node's value);
4. **activation gradients**;
5. **workspace** - what an op keeps for its backward (softmax probabilities, dropout masks, im2col buffers).

Class indices (`g.Classes()`) live in an `Array<I32>` beside it. Optimizer state (AdamW's `m` and `v`) is allocated
by the optimizer, once, in the parameters' layout. Liveness-based
reuse of regions 4 and 5 (a gradient is dead once its producer's backward has run) halves the memory of deep models;
it is a planner change only, done when a model needs it.

## 3. Operations (the closed set, first round)

| op | forward | backward |
|---|---|---|
| `MatMul(a, b, ta, tb)` | `op(a) op(b)`, one GEMM | two GEMMs with the transposes swapped |
| `Linear(x, w, b)` | `x w^T + b`: GEMM, bias in the same pass | `dx = dy w`, `dw = dy^T x`, `db` = column sums |
| `AddRow(x, b)` | every row plus `b` | `dx = dy`, `db` = column sums of `dy` |
| `Add`, `Sub`, `Mul(a, b)` | elementwise, same shape | elementwise |
| `Scale(x, c)` | `c x` | `c dy` |
| `Relu`, `LeakyRelu(a)`, `Gelu` (erf, PyTorch's default; tanh form as an option), `Sigmoid`, `Tanh` | one pass | one pass reading the saved input or output |
| `Softmax(x)` | row-wise, max-subtracted | `y (dy - rowsum(dy y))` |
| `SoftmaxCrossEntropy(logits, classes)` | mean over rows of `-log softmax[class]`, log-sum-exp | `(softmax - onehot) / n`, fused, probabilities saved in the workspace |
| `Mse(a, b)` | mean of squared differences | `2 (a - b) / count` |
| `Dropout(x, p)` (later) | mask from the graph's `Rand`, scaled by `1 / (1 - p)` | `dy` through the mask |
| `Conv2d(x, w, b, C, H, W, k, stride, pad)` (later) | im2col into the workspace, one GEMM | GEMMs and col2im |
| `LayerNorm`, `Embedding`, attention pieces (later) | | |

Fusion is the planner's job, not the layers': `Linear` then `Relu` becomes one GEMM plus one epilogue pass
(bias and activation), its backward one pass (`dpre = dy * relu'(y)`) before the GEMMs. Layers stay compositional;
speed comes from `Plan`.

## 4. Layers

A layer is a struct holding its parameters' handles; constructing it registers them with the graph, and `Apply`
records the ops. Shapes are given once, at construction:

```olang
type Dense<T> struct(g mut Graph<<T>>&, in I64, out I64, r mut Rand&) {
    W Var = g.Param(out, in, Init.KaimingUniform, r)      # PyTorch nn.Linear's defaults
    B Var = g.Param(1, out, Init.FanInUniform(in), r)
}

fn (d Dense<<T>>&) Apply(g mut Graph<<T>>&, x Var) Var { return g.Linear(x, d.W, d.B) }

type Mlp<T> struct(g mut Graph<<T>>&, r mut Rand&) {
    l1 Dense<<T>> = Dense<<T>>(g, 784, 128, r)
    l2 Dense<<T>> = Dense<<T>>(g, 128, 10, r)
}

fn (m Mlp<<T>>&) Apply(g mut Graph<<T>>&, x Var) Var { return m.l2.Apply(g, g.Relu(m.l1.Apply(g, x))) }
```

Applying a layer twice shares its parameters (their gradients add up), as in PyTorch. For the common chain there is
`Sequential`, an enum of the built-in layer kinds (`Dense`, `Relu`, `Gelu`, `Dropout`, ...) - a closed set, because
olang has no run-time interfaces yet; `any Layer&`, planned in olang, opens it later.

**Initialization** follows PyTorch's defaults so results are comparable: `nn.Linear`'s Kaiming-uniform weights
(`U(-1/sqrt(in), 1/sqrt(in))`) and fan-in-uniform biases, with He-normal and Xavier/Glorot as options, all drawn from
a seeded `Rand` passed in - so a run is reproducible from its seed.

## 5. Losses and optimizers

Losses are graph ops giving a `1 x 1` node: `SoftmaxCrossEntropy` (class indices, mean reduction - PyTorch's
`CrossEntropyLoss`), `Mse`; `g.Value(loss)` reads the number.

Optimizers update the parameter region **as one flat array** in one fused pass - PyTorch's fused/"foreach"
optimizers, Apex's multi-tensor apply - with weight decay applied to the matrices that ask for it (by default the
weights, not the biases):

- `Sgd(lr, momentum = 0, nesterov = false, weightDecay = 0)`;
- `AdamW(lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01)` with PyTorch's semantics: decoupled
  decay `p -= lr wd p`, then bias-corrected `m / (1 - b1^t)` and `v / (1 - b2^t)`, `p -= lr mhat / (sqrt(vhat) + eps)`.
  (The C version had no bias correction and divided by `v` rather than its square root.)

Each is a struct with `Step(g mut Graph<<T>>&)`; a training loop takes `opt mut <O Optimizer>&` (trait as constraint,
statically dispatched). A learning-rate schedule is a function value `fn(step I64) F64`. Gradient clipping by global
norm reads the gradient region's flat view.

## 6. Training and evaluation

```olang
fn main() ? io.IoError + os.OsError + mnist.MnistError + idx.IdxError {
    data := try mnist.Load("data/mnist", mnist.Source)
    r := rand.Rand(1)
    g := nn.Graph<F32>(128)
    x := g.Input(784)
    y := g.Classes()
    model := layers.Mlp<F32>(g, r)
    logits := model.Apply(g, x)
    loss := g.SoftmaxCrossEntropy(logits, y)
    g.Plan()
    opt := optim.AdamW<F32>(g, 1e-3)
    train := loader.Loader(data.Train, 128, 42)
    for epoch in range 10 {
        train.Epoch()
        for b in range train.Batches() {
            n := train.Fill(b, g.InputData(x), g.ClassData(y))
            g.Forward(n)
            g.Backward(loss)
            opt.Step(g)
        }
        try io.Print("epoch " $epoch ": test accuracy " $nn.Accuracy(g, logits, y, data.Test) "\n")
    }
}
```

`nn.Accuracy` runs `Forward` over a set in batches and compares each row's argmax with its class. A `Fit` helper
wraps the loop above for the common case (epochs, a schedule, a callback per epoch); the loop stays writable by hand.

## 7. Datasets (built)

- `datasets/idx` reads the IDX format (big-endian header, unsigned bytes): parsing copies nothing - the elements are
  a view of the file's bytes.
- `datasets/loader`: `Labeled` holds a set as bytes (`Features`, `Labels`, `Width`, `Classes`) with a per-feature
  `Scale` and `Shift` (default `1/255` and `0`, giving `[0, 1]`; a mean and deviation give standardization). `Loader`
  makes a seeded random order each epoch (Fisher-Yates over an index array, xoshiro256**) and `Fill(b, x, y)` writes
  batch `b`'s features as F32 rows and its classes into the caller's buffers, giving the batch's size (the last one is
  short). Nothing is allocated per batch.
- `datasets/mnist` fetches the four files with curl and gunzip into a cache directory (`data/mnist`, each step
  written to a `.part` file and renamed when whole), validates shapes and labels, and loads both sets. Any dataset of
  the MNIST family (Fashion-MNIST, KMNIST) is the same call with another address.
- **The images stay bytes in memory** - 47MB for the training set, a quarter of F32's 188MB - and become F32 as a
  batch is filled. Measured on the shared 4-core machine: the four files load at 530-830MB/s (C's `read` of the same
  files: 950MB/s), and filling an epoch of shuffled batches of 128 runs at ~4GB/s of F32 produced (1.27M samples/s).
  The data pipeline will be a rounding error beside the GEMMs.

Next: a `Dataset` trait (a constraint, as everything polymorphic in oann is) so `Loader<D>` takes other sources (CSV,
images, token streams); augmentation as function values; and filling the next batch on a task while the step runs -
`join { spawn train.Fill(b + 1, xNext, yNext) ... }` with double-buffered inputs.

## 8. Performance plan

- **GEMM is the only heavy kernel**, and it is the matrix library's: packed, blocked, multithreaded with
  `join`/`spawn`, writing into a destination with `beta` so gradient fan-in accumulates without temporaries.
- **Every other op is one pass over memory**, fused where the planner can (bias + activation into the GEMM's epilogue,
  the activation's derivative into the backward's first pass, the whole optimizer into one loop).
- **No allocation per step**, and no zeroing passes (first-write gradients).
- **Threads**: GEMM partitions its output, so results are deterministic for a given thread count; row-wise and flat
  kernels split by rows above a size threshold, through the matrix library's parallel-rows helper.
- **Target**: an epoch of the 784-128-10 MLP at the speed of a C trainer over OpenBLAS doing the same work, measured
  against a small C reference (`bench/ref`) written for the purpose - the old C code cannot serve, since it does not
  compile - and against PyTorch if it can be installed.

## 9. Testing

- Unit tests per module (`make test`), as olang `test` blocks.
- **Gradient checks**: every op's backward against central finite differences, run in `F64` (the reason the graph is
  generic over `T`), on small random shapes - the gradcheck PyTorch requires of its ops.
- The MNIST pipeline is checked against what is known about the dataset (`make data`: counts, first labels, the class
  histograms, mean 0.1307 and deviation 0.3081 of the pixels); training is checked end to end by a separate target
  (accuracy above 97% on the test set), since it needs the downloaded data.

## 10. Serialization (later)

Checkpoints as **safetensors** (8-byte header length, a JSON header naming each matrix's dtype, shape and byte range,
then the raw little-endian data): interoperable with PyTorch and Hugging Face, simple to write with std's JSON and
the floats' `Bits()`, and the parameter region is already one contiguous block. Layer parameters carry PyTorch's
names (`l1.weight`, `l1.bias`).

## 11. Shapes as type parameters (when olang has const generics)

olang will get `Matrix<T, R, C>`, each dimension a compile-time constant or known at run time. oann is shaped for it
now: **a layer's shapes are given once, at construction, and `Apply` takes only handles**, and every builder checks
shapes when it records. Then:

- `Dense<T, In, Out>` holds `W Matrix<T, Out, In>`;
- a handle carries its column count, `Var<C>` (rows are the batch, known at run time);
- `fn (d Dense<<T>, <In>, <Out>>&) Apply(g mut Graph<<T>>&, x Var<<In>>) Var<<Out>>` - a shape mismatch in a model
  becomes a compile-time error, and the record-time check becomes a type equality.

The graph's executor keeps run-time shapes (it is generic over all models), as JAX's traced programs keep shapes as
data while its Python API checks them when tracing.

## 12. Layout

```
makefile            OLANG ?= the compiler; make test, make data (fetch + check + time MNIST), make clean
rand.olang          seeded xoshiro256** (Rand: Next, Below, Float, Uniform, Normal, Shuffle)
clock.olang         a monotonic clock for timing
datasets/idx.olang      the IDX format
datasets/loader.olang   Labeled sets and the Loader
datasets/mnist.olang    fetching and loading MNIST
nn.olang            Graph, Var, the ops (forward and backward), losses, Plan          (phase 2)
layers.olang        Dense, Mlp, Sequential, initializers                              (phase 2)
optim.olang         Sgd, AdamW, schedules                                             (phase 2)
examples/           mnist_mlp.olang ...                                               (phase 2)
bench/              data.olang (the pipeline), ref/ (C over OpenBLAS), train timing   (phase 2: training)
repro/              minimal programs for olang compiler issues found here
data/, build/       downloads and build output, not in git
```

Each file is one olang module, imported by its path relative to the importing file without the extension
(`import "../datasets/mnist"`). `rand` and `clock` move to std when std grows `std/rand` and `std/time`.

## 13. What oann needs from the matrix library

oann does all its arithmetic through `Matrix<T>`; these are the forms it needs, with the signatures it would like.
Every form with a destination writes into it and allocates nothing; the operator forms (`a @ b`, `a + b`) may
allocate, for scripting. `T` is any float type, `F64` included (gradient checks), with `F32` accumulation for
`BF16`/`F16`.

```olang
# construction and views - a view is a value (rows, cols, stride, a slice of storage): making one allocates nothing
Matrix<T>(rows I64, cols I64)                                  # zero-filled, built where it lands
matrix.View(data mut Array<<T>>&, offset I64, rows I64, cols I64, stride I64 = cols) Matrix<<T>>  # over storage
                                                               # oann owns (its one arena per graph)
m.Rows(lo I64, hi I64) Matrix<<T>>                             # a row range: the short last batch
m.Cols(lo I64, hi I64) Matrix<<T>>                             # a column block by stride: attention heads
m.Row(i I64) Array<<T>>&m                                      # one row as a slice, for row-wise kernels
m.RowCount() I64, m.ColCount() I64, m.Stride() I64             # (or fields)
m[r, c], m[r, c] = v                                           # needs At/SetAt with two operands (olang E31 has one)

# BLAS level 3, the one heavy kernel: c = alpha op(a) op(b) + beta c, transposes as parameters, never in memory;
# beta = 1 accumulates a gradient's second contribution with no temporary. Packed, blocked, threaded.
matrix.Gemm(c mut Matrix<<T>>&, a Matrix<<T>>&, transA Bool, b Matrix<<T>>&, transB Bool, alpha <T> = 1.0, beta <T> = 0.0)
# later: the same with an epilogue - c = act(alpha op(a) op(b) + rowBias) - for the fused Linear + activation

# elementwise and row-broadcast, in place or into a destination (same shapes, or a 1 x C row)
c.AddRow(bias Matrix<<T>>&)                                    # every row += bias
c.ColumnSums(out mut Matrix<<T>>&, beta <T> = 0.0)             # out (1 x C) = sums over rows + beta out: db
y.AddScaled(x Matrix<<T>>&, alpha <T>)                         # y += alpha x (axpy)
m.Scale(alpha <T>), m.Fill(v <T>), m.Copy(src Matrix<<T>>&)
dst.Map(src Matrix<<T>>&, f fn(x <T>) <T>)                     # activations, a lambda inlined into the loop
dst.Map2(a Matrix<<T>>&, b Matrix<<T>>&, f fn(x <T>, y <T>) <T>)      # their backward: relu'(x) * dy
dst.Map3(a, b, c, f fn(x <T>, y <T>, z <T>) <T>)               # fused optimizer updates (p, m, v from g)

# reductions
m.RowMax(out mut Array<<T>>&), m.RowSums(out mut Array<<T>>&)  # softmax and cross-entropy pieces
m.ArgMaxRows(out mut Array<I32>&)                              # accuracy
m.Sum() <T>, m.Dot(b Matrix<<T>>&) <T>, m.Norm() <T>           # loss values, gradient clipping

# random fills, from a generator the caller seeds (reproducible runs): std/rand's Rand, shared with oann
m.FillUniform(r mut Rand&, lo <T>, hi <T>), m.FillNormal(r mut Rand&, mean <T>, sd <T>)

# threads for the kernels oann writes itself (row-wise softmax/cross-entropy, flat optimizer passes)
matrix.ParallelRows(rows I64, grain I64, body fn(lo I64, hi I64))   # join/spawn over row blocks, inline when small
```

## 14. Open questions

1. Static graph only, or also an eager mode later (PyTorch-style, every op run as it is called, a step's arena
   reclaimed at its end) for models whose structure depends on their data? Default: static only; eager when a model
   needs it.
2. The seeded generator: one `std/rand` shared by the matrix library's random fills and oann (rather than each
   having its own)? Recommended: yes - `Rand` here is ready to move.
3. After the MNIST MLP, which direction first: convolutions (im2col), transformers (attention, layer norm,
   embeddings, BF16), or spiking networks (the FPGA direction - LIF neurons as time-stepped ops)?
4. Benchmarks: write a small C trainer over OpenBLAS as the reference (the old C version does not compile), and
   install PyTorch's CPU wheel with pip (~200MB, not present) to compare against it too?
5. Should oann be laid out to be imported as a remote package (`github.com/OWNER/oann/nn`) - one module per concern
   at the top level, as above - or kept as an application repository for now?
