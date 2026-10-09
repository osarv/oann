# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

OLANG ?= /home/user/wt/oannc2/build/out

# every module with test blocks
TESTS = kernels.olang ops.olang nn.olang layers.olang optim.olang train.olang datasets/idx.olang datasets/loader.olang \
	datasets/mnist.olang datasets/text.olang generate.olang tokenizer.olang checkpoint.olang conv.olang \
	vision.olang circuit.olang agent.olang

.PHONY: test data bench epoch mnist cnn charlm lmbench lmref bpe bpelm safetensors clean

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

# an epoch's time against C over OpenBLAS (bench/epoch.sh)
epoch:
	OLANG=$(OLANG) bench/epoch.sh

clean:
	rm -rf build
