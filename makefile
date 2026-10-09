# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

OLANG ?= /home/user/wt/oannc/build/out

# every module with test blocks
TESTS = rand.olang clock.olang datasets/idx.olang datasets/loader.olang datasets/mnist.olang

.PHONY: test data bench clean c-test c-mnist

test:
	$(OLANG) -t $(TESTS)

# fetches MNIST into data/mnist (once), checks the pipeline against what is known about the dataset, and times it
data bench:
	$(OLANG) -b bench/data.olang
	./build/bench_data

clean:
	rm -rf build

# the C version this replaces, kept until the olang one trains MNIST as well (it does not compile as it stands)
CC = gcc
CFLAGS = -Wall -Werror -Wextra -Wpedantic -g -DOP_MODE_BLAS
LBINS = -lm -lopenblas -lcurl -lz
SRCS = $(filter-out test.c mnistdemo.c, $(wildcard *.c))

bin/%.o: %.c bin
	$(CC) $(CFLAGS) -c $< -o $@

bin/test%.o: %.c bin
	$(CC) $(CFLAGS) -DTEST -c $< -o $@

c-test: bin $(addprefix bin/test, $(addsuffix .o, $(basename $(SRCS))))
	$(CC) $(CFLAGS) $(filter-out bin, $^) -o bin/out $(LBINS)
	bin/out

c-mnist: bin bin/mnistdemo.o $(addprefix bin/, $(addsuffix .o, $(basename $(SRCS))))
	$(CC) $(CFLAGS) $(filter-out bin, $^) -o bin/out $(LBINS)
	bin/out

bin:
	mkdir -p bin
