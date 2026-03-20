#include <stdlib.h>
#include <stdio.h>
#include "oann.h"

int main() {
    Network n = NetworkCreate(MNIST_N_FEATURES, AdamWCreate(0.001, 0.9, 0.999, 10));
    NetworkAddLayer(n, DenseCreate(400));
    NetworkAddLayer(n, ReluCreate());
    NetworkAddLayer(n, DenseCreate(MNIST_N_LABELS));
    NetworkAddLayer(n, ReluCreate());
    NetworkSetLosser(n, SceCreate());

    Dataset mnist = MnistCreate();
    for (int i = 0; i < 100; i++) {
        printf("%f\n", NetworkTrainOneBatch(n, 1, i* 100+ 100, mnist));
    }
}
