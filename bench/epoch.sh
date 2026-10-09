#!/bin/sh
# Times an epoch of the 784-128-10 perceptron's training on MNIST: oann (on 1 and 4 threads, built for this machine -
# olang's default, B12 - and on 1 thread built for baseline x86-64, the target olang built for before) against the same
# training in C over OpenBLAS (bench/ref/mlp.c, on 1 and 4 threads). Each runs EPOCHS epochs (default 3) and reports
# its mean epoch; the variants run interleaved, ROUNDS times (default 3), and the median of the rounds is printed. Run
# from the repository's root, after "make data"; OLANG is the compiler.
set -e
OLANG=${OLANG:-/home/user/wt/oannc2/build/out}
EPOCHS=${EPOCHS:-3}
ROUNDS=${ROUNDS:-3}
$OLANG -b examples/mnist_mlp.olang >/dev/null
# the same program built for baseline x86-64 in a directory of its own, so that the two executables do not share a name
rm -rf build/baseline && mkdir -p build/baseline
cp -r nn.olang ops.olang kernels.olang conv.olang layers.olang optim.olang train.olang datasets examples build/baseline/
ln -s ../../data build/baseline/data
(cd build/baseline && $OLANG -a x86-64 -b examples/mnist_mlp.olang >/dev/null)
clang -O3 -o build/ref_mlp bench/ref/mlp.c -lopenblas -lm
epoch() { "$@" | sed -n 's/^mean epoch \([0-9.]*\) s$/\1/p'; }
median() { sort -n | awk '{ v[NR] = $1 } END { print v[int((NR + 1) / 2)] }'; }
for r in $(seq "$ROUNDS"); do
    echo "olang-1T $(epoch ./build/examples_mnist_mlp "$EPOCHS" adamw 1 1)"
    echo "olang-4T $(epoch ./build/examples_mnist_mlp "$EPOCHS" adamw 1 4)"
    echo "olang-x86-64-1T $(cd build/baseline && epoch ./build/examples_mnist_mlp "$EPOCHS" adamw 1 1)"
    echo "openblas-1T $(epoch ./build/ref_mlp "$EPOCHS" 1 1)"
    echo "openblas-4T $(epoch ./build/ref_mlp "$EPOCHS" 1 4)"
done > build/epoch.txt
echo "seconds an epoch (median of $ROUNDS rounds of $EPOCHS epochs), load average $(cut -d' ' -f1-3 /proc/loadavg):"
for v in olang-1T olang-4T olang-x86-64-1T openblas-1T openblas-4T; do
    printf '%-16s %s\n' "$v" "$(grep "^$v " build/epoch.txt | cut -d' ' -f2 | median)"
done
