# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

OLANG ?= /home/user/wt/oannc4/build/out

# every module with test blocks
TESTS = kernels.olang ops.olang nn.olang layers.olang optim.olang train.olang datasets/idx.olang datasets/loader.olang \
	datasets/mnist.olang datasets/text.olang generate.olang tokenizer.olang checkpoint.olang conv.olang \
	vision.olang sparse.olang store.olang circuit.olang board.olang agent.olang consolidator.olang quant.olang

.PHONY: test data bench epoch mnist cnn charlm lmbench lmref bpe bpelm safetensors board sparse sleep nights int8 int8lm int8bench clean

test:
	$(OLANG) -t $(TESTS)

# fetches MNIST into data/mnist (once), checks the pipeline against what is known about the dataset, and times it
data bench:
	$(OLANG) -b bench/data.olang
	./build/bench_data

# trains the 784-128-10 perceptron on MNIST and measures it on the test set after every epoch - extra arguments go
# through ARGS: epochs, optimizer (adamw, projected, sgd), seed, threads
mnist:
	$(OLANG) -b examples/mnist_mlp.olang
	./build/examples_mnist_mlp $(ARGS)

# trains a small convolutional network on MNIST (two 3x3 convolutions with ReLU and 2x2 max pooling, then a dense layer)
# - ARGS: epochs, seed, threads, the convolutions' channels C1 and C2, and "profile" for where the first epoch goes
cnn:
	$(OLANG) -b examples/mnist_cnn.olang
	./build/examples_mnist_cnn $(ARGS)

# trains the character-level transformer on tiny Shakespeare (fetched into data/shakespeare once) and generates a
# sample - ARGS: steps, seed, threads, dropout, characters to generate
charlm:
	$(OLANG) -b examples/charlm.olang
	./build/examples_charlm $(ARGS)

# the same transformer on byte-level BPE tokens (512, learned from the training part), its loss per character against
# the character model's checkpoint - ARGS: steps, seed, threads, vocabulary, tokens to generate
bpelm:
	$(OLANG) -b examples/bpelm.olang
	./build/examples_bpelm $(ARGS)

# where a transformer's training step goes, operation by operation - ARGS: steps, threads, layers, width, heads,
# context, sequences
lmbench:
	$(OLANG) -b bench/lm.olang
	./build/bench_lm $(ARGS)

# the first steps' losses against the same model in numpy (bench/ref/charlm.py) - ARGS: steps
lmref:
	$(OLANG) -b bench/lmref.olang
	./build/bench_lmref $(ARGS) > build/lmref_olang.txt
	python3 -I bench/ref/charlm.py $(ARGS) > build/lmref_numpy.txt
	paste build/lmref_olang.txt build/lmref_numpy.txt

# byte-level BPE on tiny Shakespeare: training and encoding timed, and the merges and tokens checked against an
# independent Python implementation (bench/ref/bpe.py) - ARGS: the vocabulary
bpe:
	$(OLANG) -b bench/bpe.olang
	./build/bench_bpe $(or $(ARGS),512) build/bpe_olang.txt
	python3 -I bench/ref/bpe.py data/shakespeare/input.txt $(or $(ARGS),512) build/bpe_olang.txt

# safetensors checked against a reader and writer of numpy's (bench/ref/safetensors_check.py): oann's files in every
# dtype bit by bit, and numpy's file loaded back exactly
safetensors:
	$(OLANG) -b bench/safetensors.olang
	./build/bench_safetensors write build/safetensors
	python3 -I bench/ref/safetensors_check.py build/safetensors
	./build/bench_safetensors read build/safetensors

# MNIST by a settling circuit the board can run, its lessons settled on the simulated board engine (board.olang) or in
# F32, the test set answered both ways after every epoch - ARGS: epochs, board or float, samples, hidden, batch, seed,
# rate, beta, temperature, restrain, and a window to make it spiking
board:
	$(OLANG) -b examples/mnist_board.olang
	./build/examples_mnist_board $(ARGS)

# a sparse projection's transport and contrast against a dense block's, by density (bench/sparse.olang) - ARGS: rounds,
# threads
sparse:
	$(OLANG) -b bench/sparse.olang
	./build/bench_sparse $(ARGS)

# sleep against interference (consolidator.olang): days of a flip-flop with and without nights - ARGS: nights or
# retention, seeds, days, sequences a day, repeats, hidden, window, passes, the day's step, the night's rate
sleep:
	$(OLANG) -b examples/sleep_retention.olang
	./build/examples_sleep_retention $(ARGS)

# what an agent's nights are for (examples/nights.olang, docs/settling.md 5.12): settling agents on a ring of observations
# whose answers follow a smooth rule no linear read holds, trained on a few points, tested on seen, new, noisy and partial
# observations and then living anywhere on the ring - ARGS: lives, train, after, points, hidden, block, then the agent's
# settings as name=value (sleep=1, night=, memorygain=, recallhalflife=, cues=, perlife=1, ...). The transfer and
# return tasks of examples/bandit_settle.olang measure the rest of 5.12
nights:
	$(OLANG) -b examples/nights.olang
	./build/examples_nights $(ARGS)

# the perceptron (or the CNN) trained in F32, quantized to INT8 (quant.olang) and measured both ways: test accuracy,
# the parameters' size, throughput at batch 1, 64 and 128, where the INT8 model's time goes - ARGS: mlp or cnn, epochs,
# seed, threads, timing rounds
int8:
	$(OLANG) -b examples/mnist_int8.olang
	./build/examples_mnist_int8 $(ARGS)

# the character transformer (data/shakespeare/charlm.ckpt, from make charlm) with its linear layers in INT8
# (Graph.QuantizeLinear): validation loss, a forward's time, greedy text and cached decoding against F32 - ARGS:
# characters, timing rounds, threads
int8lm:
	$(OLANG) -b examples/charlm_int8.olang
	./build/examples_charlm_int8 $(ARGS)

# the INT8 product against std/linalg's F32 product with its bias and ReLU, on oann's networks' shapes (bench/int8.olang)
# - ARGS: rounds, threads
int8bench:
	$(OLANG) -b bench/int8.olang
	./build/bench_int8 $(ARGS)

# an epoch's time against C over OpenBLAS (bench/epoch.sh)
epoch:
	OLANG=$(OLANG) bench/epoch.sh

clean:
	rm -rf build
