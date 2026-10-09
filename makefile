# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

OLANG ?= /home/user/wt/oannc2/build/out

# every module with test blocks
TESTS = ops.olang nn.olang layers.olang optim.olang datasets/idx.olang datasets/loader.olang datasets/mnist.olang

.PHONY: test data bench epoch mnist clean

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

# an epoch's time against C over OpenBLAS (bench/epoch.sh)
epoch:
	OLANG=$(OLANG) bench/epoch.sh

clean:
	rm -rf build
