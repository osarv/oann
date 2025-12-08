#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "util.h"
#include "vecarr.h"
#include "layer.h"
#include "optimizer.h"

struct network {
    int inputVecLen;
    struct vecArr features;
    struct vecArr dFeatures;
    struct list layers;
    Optimizer o;
};

void NetworkDestroy(Network n) {
    VecArrDestroy(n->features);
    VecArrDestroy(n->dFeatures);
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->destroy(l);
    }
    n->o->destroy(n->o);
    free(n);
}

Network NetworkCreate(int inputVecLen, Optimizer o) {
    Network n = MallocOrCrash(sizeof(struct network));
    n->inputVecLen = inputVecLen;
    n->layers = ListInit(sizeof(Layer));
    return n;
}

void NetworkMountDataSet(DataSet d) {
}

void NetworkAddLayer(Network n, Layer l) {
    ListAdd(&n->layers, l);
}

float* NetworkInfer(Network n) {

}

float* NetworkInferGetCopy(Network n) {
    NetworkInfer(n);
    return VecArrGetElemsCopy(losser->y);
}

void NetworkTrain(Network n) {

}

void NetworkTest(Network n) {

}
