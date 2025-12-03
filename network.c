#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "util.h"
#include "vecarr.h"
#include "layer.h"

struct network {
    int inputVecLen;
    struct vecArr features;
    struct vecArr dFeatures;
    struct list layers;
};

void NetworkDestroy(Network n) {
    VecArrDestroy(n->features);
    VecArrDestroy(n->dFeatures);
    for (int i = 0; i < n->layers.len; i++) LayerDestroy(ListGetIdx(&n->layers, i));
    free(n);
}

Network NetworkCreate(int inputVecLen) {
    Network n = MallocOrCrash(sizeof(struct network));
    n->inputVecLen = inputVecLen;
    n->layers = ListInit(sizeof(Layer));
    return n;
}

void NetworkAddLayer(Network n, Layer l) {
    ListAdd(&n->layers, l);
}
