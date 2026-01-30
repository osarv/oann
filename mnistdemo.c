#include <stdlib.h>
#include <stdio.h>
#include "oann.h"

int main() {
    Network n = NetworkCreate(MNIST_N_FEATURES, AdamWCreate(0.001, 0.9, 0.999, 0.0001));
    NetworkAddLayer(n, DenseCreate(MNIST_N_LABELS));
    NetworkAddLayer(n, ReluCreate());
    NetworkSetLosser(n, SceCreate());

    Dataset mnist = MnistCreate();
    for (int i = 0; i < 20; i++) {
        printf("%f\n", NetworkTrain(n, 100, mnist));
    }
}
