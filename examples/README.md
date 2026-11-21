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
| `charlm_sample.olang` | text from its checkpoint, through the key-value cache and by running the context again | `sample` |
| `charlm_int8.olang` | the same transformer with its linear layers in INT8: loss, a forward's time, cached decoding | `int8lm` |
| `charlm_ssm.olang` | the recurrent model (`models/ssm`: Mamba or delta-rule blocks) on tiny Shakespeare with the same recipe, generating by running the context again | |
| `bpelm.olang` | the same model on byte-level BPE tokens, loss a character against the character model's | `bpelm` |
| `tinystories.olang` | `prep`, then a GPT or an SSM of 10 million parameters on TinyStories: loss curve, tokens a second | `tinystories` |
| `graph_count.olang` | message passing (`nn/message`) over random graphs, sum against mean aggregation, on a counting and a majority node task: trained on small graphs, tested on them and on larger, denser ones | `graphcount` |
| `history_encode.olang` | `nn/history`'s encoder (one attention block, 14k parameters) against a flat perceptron (100k) over histories of 1-16 symbols, padded: the most frequent symbol, and whether the last symbol occurred before | `historyencode` |
| `mnist_learn.olang` | MNIST by a settling system of `dyn`, taught by equilibrium propagation | `mnistlearn` |
| `xor_settle.olang`, `xor_spiking.olang` | XOR by rate and by spiking `dyn` systems | `xor`, `spiking` |
| `prune_settle.olang` | a settling system learning XOR of two coordinates beside noise inputs, pruned while it runs by `dyn/learn`'s `Prune` rule toward a fraction of its weights (on a linear or cubic schedule; optionally with a skip connection into the readout), then stored as sparse rows (`System.Sparsify`): accuracy on unseen points and multiply-adds a sweep and a settle at each fraction | `prune` |
| `sleep_replay.olang` | days of a flip-flop task and then its opposite, with and without nights (`dyn/sleep`: replay through EP on the Sleep clock, the store of `dyn/store` rewritten at dawn): recall and generalization of the first task | `sleepreplay` |
| `mnist_board.olang` | MNIST through the simulated board (`board/engine`): lessons settled in F32 or on the board, the test set answered both ways, their agreement | `board` |

`sleep_replay` is the sleep-retention mechanism on `dyn`, with no agent model.
