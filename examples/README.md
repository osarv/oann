# examples

Programs that train and measure oann's models; each is the driver of a measured number, with its
arguments described at the top of the file. Build and run one from the repository's root with its make target
(`olang examples/mnist_mlp.olang 10`; the make targets below are for developers).
MNIST is fetched into `data/mnist`, tiny Shakespeare into `data/shakespeare`, TinyStories into `data/tinystories`, once.

| file | what | make |
|---|---|---|
| `mnist_mlp.olang` | the 784-128-10 perceptron (`models/mlp`) on MNIST: AdamW, the input projection, or SGD; F32 or BF16 | `mnist` |
| `mnist_cnn.olang` | the small convolutional network (`models/cnn`); resident memory and a per-op profile | `cnn` |
| `mnist_int8.olang` | either trained, quantized by `nn/quant`: accuracy dynamic and static, size, batch 1 / 64 / 128 speed | `int8` |
| `charlm.olang` | the character transformer (`models/gpt`) on tiny Shakespeare through `nn/train.Fit`, its checkpoint, a sample | `charlm` |
| `diloco_lm.olang` | the same transformer by one worker, by data parallel, or by DiLoCo worker processes sharing a directory store (`nn/dist`): validation loss against tokens and seconds, bytes a round | `diloco` |
| `charlm_sample.olang` | text from its checkpoint, through the key-value cache and by running the context again | `sample` |
| `charlm_int8.olang` | the same transformer with its linear layers in INT8: loss, a forward's time, cached decoding | `int8lm` |
| `charlm_ssm.olang` | the recurrent model (`models/ssm`: Mamba or delta-rule blocks) on tiny Shakespeare with the same recipe, generating by running the context again | |
| `bpelm.olang` | the same model on byte-level BPE tokens, loss a character against the character model's | `bpelm` |
| `tinystories.olang` | `prep`, then a GPT or an SSM of 10 million parameters on TinyStories: loss curve, tokens a second | `tinystories` |
| `evo_lm.olang` | architecture search with `evo` over tiny character LMs (`models/gpt`: depth, width, heads, window, feed-forward and norm kinds, learning rate) on tiny Shakespeare: regularized evolution, successive halving over training steps, resuming across rungs, worker processes; validation carved from the training text, the corpus's held-out tenth read once for the best | `evolm` |
| `evo_mlp.olang` | architecture search with `evo`: regularized evolution over MLPs (depth, widths, activations, learning rate) for a synthetic regression, with successive halving, weight inheritance, a checkpoint a round and worker processes; the test set read once, for the best | `evomlp` |
| `graph_count.olang` | message passing (`nn/message`) over random graphs, sum against mean aggregation, on a counting and a majority node task: trained on small graphs, tested on them and on larger, denser ones | `graphcount` |
| `history_encode.olang` | `nn/history`'s encoder (one attention block, 14k parameters) against a flat perceptron (100k) over histories of 1-16 symbols, padded: the most frequent symbol, and whether the last symbol occurred before | `historyencode` |
| `corpus.olang` | the ~300M model's pre-training corpus (`data/corpus`): `tokenizer OUT` learns a 32k BPE (digits split) from a sample of the sources or copies a teacher's tokenizer.json, `build OUT SOURCE TOKENS` streams FineWeb-Edu, FineMath or the Common Pile's Stack-Edu (Python) into filtered, deduplicated, hash-split U16 token files with a resumable state, `summary OUT` writes corpus.json | `corpus` |
| `potts_latin.olang` | `nn/potts` end to end: Latin squares learnt from solved examples (`Pairs.Screen`, `PairFitter`, `Pruned`), held-out puzzles solved by `Sampler.Anneal` and `Sampler.Focus` - accuracy and time | `potts_latin` |
| `energy_suite.olang` | the energy toolkit on generic puzzles: `Sampler.Focus` (breakout weights, `Cap`, `Hold`) on planted 3-SAT and 3-colouring, `Pairs.CutsetMove` on grid distances with walls and loops, `Factors.BySum` cage sums on Latin squares, `nn/lines` run counts with `FitRestarts` | `energybench` |
| `best_of.olang` | best-of-N by energy (`nn/bestof`, `dyn/energy.Of`): a continuous Hopfield memory on `dyn` settled from N random states, the lowest energy kept, against one settle and against any of N | `bestof` |
| `mnist_learn.olang` | MNIST by a settling system of `dyn`, taught by equilibrium propagation | `mnistlearn` |
| `xor_settle.olang`, `xor_spiking.olang` | XOR by rate and by spiking `dyn` systems | `xor`, `spiking` |
| `prune_settle.olang` | a settling system learning XOR of two coordinates beside noise inputs, pruned while it runs by `dyn/learn`'s `Prune` rule toward a fraction of its weights (on a linear or cubic schedule; optionally with a skip connection into the readout), then stored as sparse rows (`System.Sparsify`): accuracy on unseen points and multiply-adds a sweep and a settle at each fraction | `prune` |
| `sleep_replay.olang` | days of a flip-flop task and then its opposite, with and without nights (`dyn/sleep`: replay through EP on the Sleep clock, the store of `dyn/store` rewritten at dawn): recall and generalization of the first task | `sleepreplay` |
| `mnist_board.olang` | MNIST through the simulated board (`board/engine`): lessons settled in F32 or on the board, the test set answered both ways, their agreement | `board` |

`sleep_replay` is the sleep-retention mechanism on `dyn`, with no agent model.

## The energy toolkit on generic puzzles (`make energybench`)

One thread, seed 1 (`examples/potts_latin.olang` and `examples/energy_suite.olang`). "Moves" are a search's moves:
Focus moves, CutsetMove calls, or chain-sweeps for Anneal (all chains together).

| case | solved | ms a puzzle | moves a puzzle |
|---|---|---|---|
| Latin 6 x 6 learnt from 400 solved squares (Pairs, PairFitter, Pruned), Anneal 16 x 400 | 50/50 | 6-9 | 6400 chain-sweeps |
| the same, Focus 16 chains, Slack 1 | 47/50 | 5-9 | at most 20000 a chain |
| planted 3-SAT, 2000 cells, clauses 4.2 n: Focus plain | 4/4 | 56 | 174731 |
| the same, breakout (Bump 0.1, Decay 0.5) / + Cap 1 / + Hold 1 / U1 (T 0.15, Cap, Hold, Dense 8) | 4/4 each | 3.2 / 3.6 / 3.5 / 3.0 | 4291 / 5183 / 4374 / 3521 |
| planted 3-colouring, 1500 cells, mean degree 6: Focus plain (at most 400000 moves) | 1/4 | 85 | 358253 |
| the same, breakout / + Cap / + Hold / U1 | 2/4, 4/4, 4/4, 4/4 | 83 / 18 / 31 / 13 | 251283 / 66648 / 97583 / 41261 |
| distances from a source on a 10 x 10 grid, 20 % walls (loops), 40 values: CutsetMove on 3 x 3 windows | 8/8 | 227 | 825 |
| the same energy, Anneal 16 x 2000 | 0/8 | 506 | 32000 chain-sweeps |
| 6 x 6 Latin squares with 12 cages of 3 told their sums (BySum): Anneal 16 x 2000 | 1/12 | 30 | 26011 chain-sweeps |
| the same, Focus 16 chains, breakout | 12/12 | 247 | 531291 |
| run counts of 4 x 4 rows and columns, nn/lines (H 5), 4 restarts x 1500 steps (11 s once), held-out by Anneal 2 x 200 | 64/64 | 25 | 400 chain-sweeps |

Reading: breakout weights are what solve the colourings; CutsetMove's exact window moves find a distance field that
single-cell annealing cannot (each cell's value is pinned by hard neighbour constraints); the cage sums need Focus.

