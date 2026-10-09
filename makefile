# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

OLANG ?= /home/user/wt/oannc2/build/out

# every module with test blocks
TESTS = kernels.olang ops.olang nn.olang layers.olang optim.olang datasets/idx.olang datasets/loader.olang \
	datasets/mnist.olang datasets/text.olang

.PHONY: test data bench epoch mnist charlm lmbench lmref clean

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

# trains the character-level transformer on tiny Shakespeare (fetched into data/shakespeare once) and generates a
# sample - ARGS: steps, seed, threads, dropout, characters to generate
charlm:
	$(OLANG) -b examples/charlm.olang
	./build/examples_charlm $(ARGS)

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

# an epoch's time against C over OpenBLAS (bench/epoch.sh)
epoch:
	OLANG=$(OLANG) bench/epoch.sh

clean:
	rm -rf build
