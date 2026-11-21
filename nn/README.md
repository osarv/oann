# nn

What models are built from and trained with, on the autodiff graph (`ad/`).

- `layers.olang` - layers as plain data (handles and sizes) with a `New` on a graph that registers the parameters and an
  `Apply` that records the operations: `Dense`, `Embedding` (also a tied output head), `Norm` (layer or RMS), `Mlp`,
  `Conv`, `SelfAttention` (rotary positions optional), `AttentionBlock` (pre-normalized, a GELU or gated SwiGLU `Feed`,
  dropout), `Transformer` (GPT-2's decoder), and the scan-based residual blocks `SsmBlock` (Mamba's selective scan over
  the graph's `SelectiveScan`) and `DeltaBlock` (the gated delta rule). Starting values are PyTorch's for `Dense` and
  `Conv`, GPT-2's for the transformer.
- `optim.olang` - `AdamW` and `Sgd` over ONE FLAT REGION (two tensors laid out alike: a graph's
  `Params()`/`ParamGrads()`, a settling system's, a BF16 graph's `Masters()` with its BF16 gradients), the runs of the
  region that decay (`Spans(g)`), `ClipNorm`, the schedules `WarmupCosine` and `WarmupLinear`, and `Projector`, the
  delta-rule input projection that stands in for weight decay; `Muon` (momentum, then Newton-Schulz orthogonalization of each
  2-D parameter's step - `MuonMatrices(g)` lists them, the other elements go to AdamW at `adamRate`; `Orthogonalize` is
  the Newton-Schulz step on its own) and `Ema`, a weight average (`Update` after each step, `Swap` to evaluate with the
  average and `Swap` back).
- `train.olang` - the loops: `Epoch`/`Evaluate` for a classifier over a `loader.Loader`; `LmStep`/`LmEvaluate` for a
  language model over any batch source with `Random` and `At` (data/text's `Windows`, data/tokens' `Tokens`); `Fit`, a
  whole run with reports, a record through io/runlog and checkpoints; `Save`/`Resume`, safetensors files holding the
  parameters, the optimizer's state and the step, which resume a run bit for bit.
- `generate.olang` - `Sample` and the generators: `CachedTokens`/`Cached` through a decoding graph's key-value cache, `CachedBatch` (a batch of sequences decoded at once on a graph planned to decode Batch sequences: a forward of one position a sequence a step, each sequence its own cache - several samples of a prompt as rows),
  `RerunTokens`/`Rerun` running the context again, on token ids and on text.

- `message.olang` - message passing over a sparse graph: `Edges` (each node's incoming edges, compressed rows, set afresh from an edge list each batch, or by `SetDense` from a batch of dense adjacency matrices - entry (v, u) an edge u -> v of that weight - without allocating), `Passing` (an operation of a `Graph<F32>` through `Custom`: each node's message the `Agg.Sum` or `Agg.Mean` of its in-neighbours' rows, its backward the transpose), and `Layer` (act(x Self^T + message Msg^T + B)). A batch of graphs is one graph of disjoint parts, a part's nodes a run of rows.

- `history.olang` - `Encoder`: a small self-attention encoder over a variable-length history (up to `Ctx` entries of features): a projection, a learned position each, one causal pre-normalized `AttentionBlock`, a final norm, then a pooling by weights (`Weights`: the mean of the valid positions, `Pool.Mean`, or the last valid one, `Pool.Last`). Right-padded histories are exact: causal attention never reads the padding, so the encoding and every gradient do not depend on it (`Rows`, `Pooled` apart).

- `bestof.olang` - best-of-N selection by a model's own score: `BestOf<T>` keeps, row by row, the candidate with the lowest (`Order.Lowest`, an energy or a loss) or highest (`Order.Highest`, a likelihood or a verifier's score) score - `Offer`, `OfferRows`, `Search` (the candidates made by a function), `Rank` (proposed by one function, scored by another: one model proposes, another's energy ranks), `Best`, `Score`, `Index`; NaN never wins, the first of equals is kept; allocation-free.

- `quant.olang` - INT8 post-training quantization for inference over kernels/gemm's `GemmI8`, on tensors and arrays alone: `Weights` (symmetric per output channel), `Range` (dynamic or calibrated activation ranges, U8 with a zero point), `Model` (dense layers, convolutions (their windows read from the quantized images by `GemmI8Left`, no patch matrix) and max pooling from weight tensors and biases - `DenseOf`, `ConvOf`, `Pool` - run on plain arrays by `Forward`; `Calibrate` and `Static`; `Bytes`).
- `quantgraph.olang` - `quant` read from the autodiff graph: `Dense(q, g, d, relu)`, `Conv(q, g, c, relu)` and `Mlp(batch, g, net)` from a trained graph's values, and `Quantize(g)` / `Dequantize(g)` / `LinearBytes(g)` for the graph's own hook (every `Linear` node in INT8 - a transformer's projections and perceptrons, its key-value decoding unchanged). Apart from `quant` so that a program quantizing without the graph does not compile the graph (whose size makes the build link by ThinLTO, which costs the kernels their inlining: repro/crossinline, fixed in olang and removed 2026-11-21).

A step, an optimizer step and a training step allocate nothing: each has a test that runs 20 steps after the first and
asserts the process's resident memory did not grow.

Test with `olang test nn/FILE.olang`; `make test` runs them all. `make charlm` trains the character transformer
through them (`examples/charlm.olang`, the model is `models/gpt`).
