# nn

What models are built from and trained with, on the autodiff graph (`ad/`).

- `layers.olang` - layers as plain data (handles and sizes) with a `New` on a graph that registers the parameters and an
  `Apply` that records the operations: `Dense`, `Embedding` (also a tied output head), `Norm` (layer or RMS), `Mlp`,
  `Conv`, `SelfAttention` (rotary positions optional; `window` W > 0 for sliding-window attention, the graph's `Attention(..., window)`; `gate` for a per-head output gate, head h's output times sigmoid(W x)_h with W [heads, D] - `HeadGate` applies one to any [b, t, D]), `AttentionBlock` (pre-normalized, a GELU or gated SwiGLU `Feed`,
  dropout), `Transformer` (GPT-2's decoder), and the scan-based residual blocks `SsmBlock` (Mamba's selective scan over
  the graph's `SelectiveScan`) and `DeltaBlock` (the gated delta rule: the graph's `DeltaRule`, chunked in the WY form for training, the recurrent form a position at a time while decoding, its state carried between Forward calls). Starting values are PyTorch's for `Dense` and
  `Conv`, GPT-2's for the transformer.
- Packed documents (several documents to a sequence): `DocStarts(g, tokens, sep, docs, keep)` fills, from the batch's
  tokens and a separator that begins each document, `docs` (a `Tokens(t)`, each position's document start) for
  `SelfAttention.Apply(g, x, docs)` - the graph's `Attention(..., docs)`, attention within a document - and `keep` (an
  `Input(t, 1)`, zero where a document starts) for `DeltaBlock.Apply(g, x, keep)`, whose gate it multiplies so the state
  starts from nothing at each document.
- `ngram.olang` - input tables added to the embedding, lookups only: `Ngram` of order 1 (the previous token's code, a
  table indexed by x_{t-1}, its last row for the first position), 2 or 3 (a table of `Buckets` rows indexed by a hash of
  x_{t-1}, x_t or x_{t-2}, x_{t-1}, x_t). `New` registers the table and a `Tokens()` input of row indices, `Fill` computes
  those from the batch's tokens (while decoding, the new positions; `before` carries the two tokens before a sequence's
  first position), `Apply` gives the rows [b, t, Dim]: `h = g.Add(emb.Apply(g, tok), ng.Apply(g))`.
- `optim.olang` - `AdamW` and `Sgd` over ONE FLAT REGION (two tensors laid out alike: a graph's
  `Params()`/`ParamGrads()`, a settling system's, a BF16 graph's `Masters()` with its BF16 gradients), the runs of the
  region that decay (`Spans(g)`), sparse tables stepped by their touched rows (`o.Sparse(g)` after `g.Sparse(table)`:
  lazy Adam, exact where a row is touched every step; `Muon.Sparse` the same for its AdamW part), `ClipNorm`, the schedules `WarmupCosine` and `WarmupLinear`, and `Projector`, the
  delta-rule input projection that stands in for weight decay; `Muon` (momentum, then Newton-Schulz orthogonalization of each
  2-D parameter's step - `MuonMatrices(g)` lists them, the other elements go to AdamW at `adamRate`; `Orthogonalize` is
  the Newton-Schulz step on its own) and `Ema`, a weight average (`Update` after each step, `Swap` to evaluate with the
  average and `Swap` back).
- `train.olang` - the loops: `Epoch`/`Evaluate` for a classifier over a `loader.Loader`; `LmStep`/`LmEvaluate` for a Data parallel: `AverageGrads` and `BroadcastParams` over replicas - one single-thread graph a thread at its share of the batch, run at once (a join), the gradients averaged into the first, the optimizer stepping it, its parameters copied back; for small models whose products split poorly across threads.
  language model over any batch source with `Random` and `At` (data/text's `Windows`, data/tokens' `Tokens`); `Fit`, a
  whole run with reports, a record through io/runlog and checkpoints; `Save`/`Resume`, safetensors files holding the
  parameters, the optimizer's state and the step, which resume a run bit for bit.
- `dist.olang` - one model trained by workers that share no network, only a store of named blobs: DiLoCo. Each worker
  trains its own copy for H steps with its own optimizer, `Sync` hands in its delta (global - local) and takes back the
  outer step the coordinator (worker 0) took on the average - Nesterov momentum, DiLoCo's 0.7 and 0.9. `Store` is the
  transport (`Put`/`Get` of named blobs): `DirStore` a directory, `HttpStore` PUT/GET under a base URL. Deltas and updates
  go as F32, BF16, Int8 (blocks of 256) or TopK (error feedback kept in the worker's model); `Fragments` > 1 streams
  (a fragment a round); a worker missing the coordinator's deadline is left out and folded in a round late; `Begin`
  joins a run from its newest `state.<R>` (a safetensors checkpoint: the flat region and the outer momentum) and the
  updates after it. README "Distributed training" has the blob names and the HTTP contract.
- `generate.olang` - `Sample` and the generators: `CachedTokens`/`Cached` through a decoding graph's key-value cache, `CachedBatch` (a batch of sequences decoded at once on a graph planned to decode Batch sequences: a forward of one position a sequence a step, each sequence its own cache - several samples of a prompt as rows),
  `RerunTokens`/`Rerun` running the context again, on token ids and on text.

- `message.olang` - message passing over a sparse graph: `Edges` (each node's incoming edges, compressed rows, set afresh from an edge list (`Set`), adjacency lists (`SetAdjacency`) or dense matrices (`SetDense`) each batch, or by `SetDense` from a batch of dense adjacency matrices - entry (v, u) an edge u -> v of that weight - without allocating), `Passing` (an operation of a `Graph<F32>` through `Custom`: each node's message the `Agg.Sum` or `Agg.Mean` of its in-neighbours' rows, its backward the transpose), and `Layer` (act(x Self^T + message Msg^T + B)). A batch of graphs is one graph of disjoint parts, a part's nodes a run of rows.

- `history.olang` - `Encoder`: a small self-attention encoder over a variable-length history (up to `Ctx` entries of features): a projection, a learned position each, one causal pre-normalized `AttentionBlock`, a final norm, then a pooling by weights (`Weights`: the mean of the valid positions, `Pool.Mean`, or the last valid one, `Pool.Last`). Right-padded histories are exact: causal attention never reads the padding, so the encoding and every gradient do not depend on it (`Rows`, `Pooled` apart).

- `bestof.olang` - best-of-N selection by a model's own score: `BestOf<T>` keeps, row by row, the candidate with the lowest (`Order.Lowest`, an energy or a loss) or highest (`Order.Highest`, a likelihood or a verifier's score) score - `Offer`, `OfferRows`, `Search` (the candidates made by a function), `Rank` (proposed by one function, scored by another: one model proposes, another's energy ranks), `Best`, `Score`, `Index`; NaN never wins, the first of equals is kept; allocation-free.
- `ternary.olang` - ternary weights while training (BitNet b1.58): `Quantizer` (an operation: a weight matrix's rows quantized by absmean to -1, 0, +1 times their scale, its gradient passed straight through), `Linear` (a dense layer reading its latent weights through it; `Export` gives kernels/ternary's form for inference, the same product)
- `chain.olang` - a chain of hidden units summed out exactly by forward-backward (`Chain`: visible cells of K values, a hidden state of H values per position, tables U, V shared by position and indexed by its context): `FreeEnergy`, `Conditionals` (every cell's free energy at each value, one forward and one backward pass), `PseudoGrad` (the pseudolikelihood loss and its exact gradient, reverse mode through forward-backward) with `Step` (Adam), and `Anneal` (ordered Gibbs scans with the chain summed out). Learns a latent running state nobody labels - the carry of decimal addition from 1..4 digits, exact at 12.
- `diffusion.olang` - discrete diffusion over cells of K values (`Diffusion`, `Kind.Masked` or `Kind.Uniform`): `Corrupt` makes a denoiser's training example at level t, `Sample` runs the reverse process from noise with any denoiser giving p(x0 | x_t) (exact posterior steps); with `bestof.Rank` and an energy as the score it is a propose-and-rank pipeline - diffusion proposes, the energy picks.
- `potts.olang` - a pairwise (Potts) energy over a sparse graph of K-valued cells (`Potts`: a shared K x K table W, unary terms linear in each cell's features, `Factors` - k-ary factor tables - and `Hidden` - binary hidden units, block-sampled; `Colouring`; `ClueFeatures`, a constraint problem's cell features: clue one-hot and degree), fitted by pseudolikelihood (`PseudoGrad`, `Symmetric`; `Fitter` - Add a batch's examples, Step by Adam; local rules on the same tables: `Stats` - the energy's derivative at an assignment - `AddContrast`, `CD` (from the data, from noise, or persistent) and `EP` (free, then nudged by +-beta through the sampler's `Bias`)) and searched by many chains at once (`Sampler`: `Colour` the graph's colour classes, `Start`, `Anneal` - annealed chromatic Gibbs to a zero-temperature finish, each chain stopping on its own, `Schedule` - `Focus`, a focused heat bath over violated factors and edges with optional growing and decaying constraint weights (`FocusSchedule`), and `Langevin`, the relaxed search on softmax logits; `Best`), over message.Edges `Pairs` - a table of its own for each listed pair of cells and a bias per cell (`All`, or `Screen`: the pairs of highest mutual information over solved examples), fitted by `PairFitter` (pseudolikelihood, Adam, L1 soft thresholds), thinned by `Pruned` (couplings with row and column means taken out), and solved exactly by `ForestMin` (min-sum dynamic programming when the free cells' pair graph is a forest, else `NotForest.Cycle`).
- `seqsettle.olang` - a settle as a sequence layer (`SettleSeq`, a graph operation): each token's state K sweeps of s = tanh(W s + b_t), W block-diagonal in heads, the drive b_t = u_t plus the previous state rotated pair by pair by phi_t and scaled by gam_t; cold or warm, with or without the carry; forward and exact backward through every sweep, the batch's sequences as vector lanes; `bench/seqsettle.olang` times it

- `quant.olang` - INT8 post-training quantization for inference over kernels/gemm's `GemmI8`, on tensors and arrays alone: `Weights` (symmetric per output channel), `Range` (dynamic or calibrated activation ranges, U8 with a zero point), `Model` (dense layers, convolutions (their windows read from the quantized images by `GemmI8Left`, no patch matrix) and max pooling from weight tensors and biases - `DenseOf`, `ConvOf`, `Pool` - run on plain arrays by `Forward`; `Calibrate` and `Static`; `Bytes`).
- `quantgraph.olang` - `quant` read from the autodiff graph: `Dense(q, g, d, relu)`, `Conv(q, g, c, relu)` and `Mlp(batch, g, net)` from a trained graph's values, and `Quantize(g)` / `Dequantize(g)` / `LinearBytes(g)` for the graph's own hook (every `Linear` node in INT8 - a transformer's projections and perceptrons, its key-value decoding unchanged). Apart from `quant` so that a program quantizing without the graph does not compile the graph (whose size makes the build link by ThinLTO, which costs the kernels their inlining: repro/crossinline, fixed in olang and removed 2026-11-21).

A step, an optimizer step and a training step allocate nothing: each has a test that runs 20 steps after the first and
asserts the process's resident memory did not grow.

Test with `olang test nn/FILE.olang`; `make test` runs them all. `make charlm` trains the character transformer
through them (`examples/charlm.olang`, the model is `models/gpt`).

## Energy models over discrete cells (`potts.olang`)

A model gives every assignment of K-valued cells an energy; low energy means the cells obey the rule it learnt.
Answers are found by searching for low energy, and new rules are learnt from solved examples. The pieces:

- **Energies.** `Potts` has one K x K table W shared by the edges of a graph (each edge with a weight), per-cell terms
  linear in the cells' features (B), and optional `Factors` (a table over several cells) and `Hidden` units. `Pairs`
  gives every listed pair its own K x K table plus a bias per cell. `Pairs.Model()` turns it into a Potts model whose
  factors are the pairs, and `Pairs.Bias(clamp, hold)` gives the matching sampler bias. A cell's local energies, the
  energy of each of its values with the others held, are what every search and fitter reads.
- **Learning from solved examples.**
  - Pseudolikelihood: each cell's value is predicted from its local energies given the others. `Fitter` does this for
    `Potts`; `PairFitter` does it for `Pairs`, with Adam and L1 soft thresholds that drive an uncoupled pair's table
    to zero.
  - Block pseudolikelihood (`Fitter.AddBlocks`) predicts each factor's whole scope jointly from the rest; cells in
    no factor are predicted on their own. Per-cell pseudolikelihood falls into a redundancy trap when cells are
    pinned by several constraints: each cell is explained by its other constraints, so none is learnt. A block cannot
    be explained away. A factor's table can be indexed by the sum of its cells' values (`Factors.BySum`), K x arity
    entries instead of K^arity.
  - `Pairs.Screen` picks which pairs to fit: those whose values share the most mutual information across the
    examples.
  - `Pairs.Pruned` drops the pairs whose coupling stays small. The coupling is a table's largest entry once its row
    and column means are taken out, because the biases could absorb those means.
  - The local rules (`Stats`, `CD`, `EP`) fit the same `Potts` tables by contrasting the energy's derivatives at the
    data with those at the sampler's states.
- **`Sampler.Anneal`** runs many chains at once, 16 to a vector lane. A sweep visits the cells colour class by colour
  class: cells of one class share no edge or factor, so they can be redrawn together. Each cell is redrawn from
  softmax(-local / T) as T falls, then zero-temperature sweeps run until nothing moves. Clamped cells (the givens) go
  in through `Bias`. Each chain's answer is in `X`, and `Best()` names the chain of lowest energy.
- **`Sampler.Focus`** is the focused counterpart, for energies made of constraints.
  - A move picks a violated constraint at random and redraws one of its cells. A constraint is violated when its
    energy is more than `Slack` above its table's lowest.
  - Optional weights (`Bump`, `Decay`) grow on constraints the search keeps breaking.
  - Each chain stops once no constraint is violated. Each cell's local energies are kept up to date by differences
    on every move.
  - Only cells in violated constraints move, so start the chains at the givens. A hold must outweigh every pair a
    given could break.
- **`Pairs.ForestMin(clamp, x)`** is exact when the free cells' pairs form a forest (edges = cells - components). It
  runs min-sum from the leaves up, then reads the best values back down. Otherwise it returns `NotForest.Cycle`.
- **`Pairs.CutsetMove(x, region, count, r)`** re-solves a region of cells exactly, with every other cell held. Held
  cells are cut from the region at random until its free cells have at most one cycle. One cell on that cycle is then
  conditioned on each of its values, and the forest left is solved by `ForestMin`. x takes the best completion, so
  the energy never rises; repeated over windows or random regions, it is a large-neighbourhood search.

- **Rules over whole lines (`lines.olang`).** A `Lines` factor gives each line of cells (a grid's rows and columns,
  any sequences) a learned hidden chain of H states. Its tables are shared by every line: transitions A[h'][h][x] and
  an end table Z[clue][h] read at the line's clue, such as its count. The chain is summed out exactly by
  forward-backward, so a line's energy is its free energy.
  - `PseudoGrad` is exact per-cell pseudolikelihood. A cell's conditional reads only the free energies of the lines
    through it, so the gradient is those lines' expected counts at each of its values.
  - `Anneal` searches the summed free energy.
  - The states' meaning, such as a run counter, is found from only some random starts, and the training loss cannot
    tell which. `FitRestarts` therefore keeps the restart with the best score on a validation function the caller
    supplies.

`examples/potts_latin.olang` runs the whole pipeline: Latin squares, then screen, fit and prune, then held-out puzzles
solved by `Anneal` and by `Focus`.
