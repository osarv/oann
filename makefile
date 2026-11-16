# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

# the compiler: olang's build/olang (its makefile's "make"), olang checked out beside oann by default
OLANG ?= ../olang/build/olang

# every module with test blocks
TESTS = tensor.olang kernels/mat.olang kernels/gemm.olang kernels/elem.olang kernels/attention.olang kernels/scan.olang kernels/sparse.olang kernels/conv.olang \
	ad/graph.olang tools/gradcheck.olang nn/layers.olang nn/optim.olang nn/train.olang nn/generate.olang nn/quant.olang nn/quantgraph.olang \
	models/mlp.olang models/cnn.olang models/gpt.olang models/ssm.olang tokenizer.olang \
	data/idx.olang data/loader.olang data/mnist.olang data/text.olang data/tokens.olang io/checkpoint.olang io/runlog.olang \
	dyn/neuron.olang dyn/conn.olang dyn/system.olang dyn/energy.olang dyn/learn.olang dyn/store.olang dyn/sleep.olang dyn/bridge.olang models/hybrid.olang board/engine.olang

.DEFAULT_GOAL := help

.PHONY: help test mlpad arenaad tinystories data bench epoch mnist cnn charlm lmbench lmref bpe bpelm board int8 int8lm int8bench storagebench int8cnnbench int8decodebench xor spiking prune hybridbench sleepreplay mnistlearn sample clean

# prints this list: every target with the comment above it (its ARGS included) - the default, runs nothing
help:
	@awk '/^#/ { sub(/^# ?/, ""); c = c (c == "" ? "" : " ") $$0; next } \
	  /^[a-z][a-z0-9 ]*:/ { if (c != "") { n = $$0; sub(/:.*/, "", n); printf "%-16s %s\n", n, c } } \
	  { c = "" }' $(MAKEFILE_LIST)

# every module's tests (takes minutes) - OLANG=path names another compiler
test:
	$(OLANG) test $(TESTS)

# M1 on ad/graph: the 784-128-10 perceptron on MNIST, each epoch timed - ARGS: epochs, threads, seed, and bf16
mlpad:
	$(OLANG) build -output build/bench_mlp_ad bench/mlp_ad.olang
	./build/bench_mlp_ad $(ARGS)

# M9 on ad/graph: round 2's transformer's arena with activations reused and apart, and the resident memory per step -
# ARGS: steps, threads, and bf16
arenaad:
	$(OLANG) build -output build/bench_arena_ad bench/arena_ad.olang
	./build/bench_arena_ad $(ARGS)

# fetches MNIST into data/mnist (once), checks the pipeline against what is known about the dataset, and times it
data bench:
	$(OLANG) build -output build/bench_data bench/data.olang
	./build/bench_data

# trains the 784-128-10 perceptron on MNIST and measures it on the test set after every epoch - extra arguments go
# through ARGS: epochs, optimizer (adamw, projected, sgd), seed, threads
mnist:
	$(OLANG) build -output build/examples_mnist_mlp examples/mnist_mlp.olang
	./build/examples_mnist_mlp $(ARGS)

# trains a small convolutional network on MNIST (two 3x3 convolutions with ReLU and 2x2 max pooling, then a dense layer)
# - ARGS: epochs, seed, threads, the convolutions' channels C1 and C2, and "profile" for where the first epoch goes
cnn:
	$(OLANG) build -output build/examples_mnist_cnn examples/mnist_cnn.olang
	./build/examples_mnist_cnn $(ARGS)

# M2 and M3: the character-level transformer (models/gpt) on tiny Shakespeare (fetched into data/shakespeare once)
# through nn/train's Fit, its checkpoint in data/shakespeare, and a sample through the key-value cache - ARGS: steps,
# seed, threads, dropout, characters to generate, and bf16 last for mixed precision
charlm:
	$(OLANG) build -output build/examples_charlm examples/charlm.olang
	./build/examples_charlm $(ARGS)

# M13: a GPT and an SSM of 10 million parameters on TinyStories (examples/tinystories.olang) - first ARGS="prep", then
# ARGS="gpt TOKENS" or "mamba TOKENS" (context, sequences, seed, threads, bf16 after)
tinystories:
	$(OLANG) build -output build/examples_tinystories examples/tinystories.olang
	./build/examples_tinystories $(ARGS)

# the same transformer on byte-level BPE tokens (512, learned from the training part), its loss per character against
# the character model's checkpoint - ARGS: steps, seed, threads, vocabulary, tokens to generate
bpelm:
	$(OLANG) build -output build/examples_bpelm examples/bpelm.olang
	./build/examples_bpelm $(ARGS)

# where a transformer's training step goes, operation by operation - ARGS: steps, threads, layers, width, heads,
# context, sequences
lmbench:
	$(OLANG) build -output build/bench_lm bench/lm.olang
	./build/bench_lm $(ARGS)

# the first steps' losses against the same model in numpy (bench/ref/charlm.py) - ARGS: steps
lmref:
	$(OLANG) build -output build/bench_lmref bench/lmref.olang
	./build/bench_lmref $(ARGS) > build/lmref_olang.txt
	python3 -I bench/ref/charlm.py $(ARGS) > build/lmref_numpy.txt
	paste build/lmref_olang.txt build/lmref_numpy.txt

# byte-level BPE on tiny Shakespeare: training and encoding timed, and the merges and tokens checked against an
# independent Python implementation (bench/ref/bpe.py) - ARGS: the vocabulary
bpe:
	$(OLANG) build -output build/bench_bpe bench/bpe.olang
	./build/bench_bpe $(or $(ARGS),512) build/bpe_olang.txt
	python3 -I bench/ref/bpe.py data/shakespeare/input.txt $(or $(ARGS),512) build/bpe_olang.txt

# MNIST by a settling system the board can run, its lessons settled on the simulated board engine (board/engine) or in
# F32, the test set answered both ways after every epoch - ARGS: epochs, board or float, samples, hidden, batch, seed,
# rate, beta, temperature
board:
	$(OLANG) build -output build/examples_mnist_board examples/mnist_board.olang
	./build/examples_mnist_board $(ARGS)

# M14 (docs/redesign.md 4.3): models/hybrid's transformer with a settle block on tiny Shakespeare, trained twice - every
# settle from rest, then from the amortizer's guess - and the sweeps a settle takes in each (bench/hybrid.olang) - ARGS:
# steps, seed, units, gradient (exact, jfb, ep), tolerance, kappa, report interval
hybridbench:
	$(OLANG) build -output build/bench_hybrid bench/hybrid.olang
	./build/bench_hybrid $(ARGS)

# XOR by rate neurons on dyn, taught by equilibrium propagation (examples/xor_settle.olang)
xor:
	$(OLANG) build -output build/examples_xor_settle examples/xor_settle.olang
	./build/examples_xor_settle $(ARGS)

# a settling system pruned while it runs, then stored sparse (examples/prune_settle.olang) - ARGS: score, fraction kept,
# pace, regrowth, hidden, noise inputs, lessons, lessons between ticks, seed, rate
prune:
	$(OLANG) build -output build/examples_prune_settle examples/prune_settle.olang
	./build/examples_prune_settle $(ARGS)

# XOR by spiking neurons on dyn (examples/xor_spiking.olang)
spiking:
	$(OLANG) build -output build/examples_xor_spiking examples/xor_spiking.olang
	./build/examples_xor_spiking $(ARGS)

# sleep replay on dyn (examples/sleep_replay.olang) - ARGS: seeds, days, patterns a day, hidden, passes, rate, context,
# store cells, active cells
sleepreplay:
	$(OLANG) build -output build/examples_sleep_replay examples/sleep_replay.olang
	./build/examples_sleep_replay $(ARGS)

# the same network on dyn (dyn/learn's EP lessons; docs/redesign.md 4.3, M8) - ARGS: epochs, samples, hidden, batch,
# seed, Adam's rate, beta, threads
mnistlearn:
	$(OLANG) build -output build/examples_mnist_learn examples/mnist_learn.olang
	./build/examples_mnist_learn $(ARGS)

# text from the checkpoint make charlm wrote (examples/charlm_sample.olang)
sample:
	$(OLANG) build -output build/examples_charlm_sample examples/charlm_sample.olang
	./build/examples_charlm_sample $(ARGS)

# the perceptron (or the CNN) trained in F32, quantized to INT8 (nn/quant.olang, M12) and measured both ways: test accuracy,
# the parameters' size, throughput at batch 1, 64 and 128 - ARGS: mlp or cnn, epochs,
# seed, threads, timing rounds
int8:
	$(OLANG) build -output build/examples_mnist_int8 examples/mnist_int8.olang
	./build/examples_mnist_int8 $(ARGS)

# the character transformer (data/shakespeare/charlm.ckpt, from make charlm) with its linear layers in INT8
# (nn/quantgraph's Quantize): validation loss, a forward's time, greedy text and cached decoding against F32 - ARGS:
# characters, timing rounds, threads
int8lm:
	$(OLANG) build -output build/examples_charlm_int8 examples/charlm_int8.olang
	./build/examples_charlm_int8 $(ARGS)

# the INT8 product against kernels/gemm's F32 product with its bias and ReLU, on oann's networks' shapes (bench/int8.olang)
# - ARGS: rounds, threads
int8bench:
	$(OLANG) build -output build/bench_int8 bench/int8.olang
	./build/bench_int8 $(ARGS)

# F32 activations times weights stored in F32, BF16 or INT8 through one Gemm (bench/storage.olang) - ARGS: rounds, threads
storagebench:
	$(OLANG) build -output build/bench_storage bench/storage.olang
	./build/bench_storage $(ARGS)

# the INT8 CNN of nn/quant against the graph's F32 forward, as drawn, at batch 1, 64, 128 (bench/int8cnn.olang) - ARGS:
# rounds, threads
int8cnnbench:
	$(OLANG) build -output build/bench_int8cnn bench/int8cnn.olang
	./build/bench_int8cnn $(ARGS)

# the character GPT decoding through the key-value cache, F32 against INT8 (bench/int8decode.olang) - ARGS: characters,
# rounds, threads, and int8 to time INT8 alone (for olang profile)
int8decodebench:
	$(OLANG) build -output build/bench_int8decode bench/int8decode.olang
	./build/bench_int8decode $(ARGS)

# an epoch's time against C over OpenBLAS (bench/epoch.olang)
epoch:
	$(OLANG) build -output build/bench_epoch bench/epoch.olang
	OLANG=$(OLANG) ./build/bench_epoch

# removes build/
clean:
	rm -rf build
