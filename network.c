#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "util.h"
#include "vecarr.h"
#include "layer.h"
#include "losser.h"
#include "optimizer.h"
#include "dataset.h"

struct network {
    int inputVecLen;
    int batchSize;
    struct vecArr features;
    struct vecArr dFeatures; //needed for generic layer backward calling
    struct vecArr labels;
    struct list layers;
    Losser lsr;
    Optimizer o;
};

static void networkUnmount(Network n) {
    VecArrDestroy(n->features);
    VecArrDestroy(n->dFeatures);
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->unmount(l);
    }
    n->lsr->unmount(n->lsr);
    VecArrDestroy(n->labels);
}

static void networkMount(Network n) {
    n->features = VecArrCreate(n->inputVecLen, n->batchSize);
    n->dFeatures = VecArrCreateSameDim(n->features);
    struct vecArr x = n->features;
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->mount(l, x);
        x = l->y;
    }
    n->lsr->mount(n->lsr, x);
    n->labels = VecArrCreate(n->lsr->y.vecLen, n->batchSize);
}

void NetworkDestroy(Network n) {
    if (n->batchSize != 0) networkUnmount(n);
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->destroy(l);
    }
    n->lsr->destroy(n->lsr);
    n->o->destroy(n->o);
    free(n);
}

Network NetworkCreate(int inputVecLen, Optimizer o) {
    Network n = CallocOrCrash(sizeof(struct network)); //zero batch size
    n->inputVecLen = inputVecLen;
    n->layers = ListInit(sizeof(Layer));
    n->o = o;
    return n;
}

void NetworkAddLayer(Network n, Layer l) {
    ListAdd(&n->layers, l);
}

static void networkSetBatchSize(Network n, int batchSize) {
    if (n->batchSize == batchSize) return;
    networkUnmount(n);
    n->batchSize = batchSize;
    networkMount(n);
}

//assumes proper mounting
static void networkForward(Network n, struct vecArr x) {
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->forward(l, x);
        x = l->y;
    }
    n->lsr->forward(n->lsr, x);
}

//assumes proper mounting
static void networkBackward(Network n, struct vecArr labels) {
    Layer prevL = ListGetIdx(&n->layers, n->layers.len -1);
    n->lsr->backward(n->lsr, prevL->y, labels, prevL->dy);
    Layer l = prevL;
    for (int i = n->layers.len -1; i >= 1; i--) {
        prevL = ListGetIdx(&n->layers, i -1);
        l->backward(l, prevL->y, prevL->dy);
        prevL = l;
    }
    l->backward(l, n->features, n->dFeatures);
}

void NetworkTrain(Network n, DataSet d) {
    networkSetBatchSize(n, d->trainBatchSize);
    for (int i = 0; i < d->nTrainBatches; i++) {
        struct vecArr x = d->getTrainFeatures(d, n->features, i);
        struct vecArr labels = d->getTrainLabels(d, n->features, i);
        networkForward(n, x);
        networkBackward(n, labels);
    }
}

float NetworkTest(Network n, DataSet d) {
    networkSetBatchSize(n, d->testBatchSize);
    float loss = 0;
    for (int i = 0; i < d->nTestBatches; i++) {
        struct vecArr x = d->getTestFeatures(d, n->features, i);
        struct vecArr labels = d->getTestLabels(d, n->labels, i);
        networkForward(n, x);
        Layer lastL = ListGetIdx(&n->layers, n->layers.len -1);
        loss += n->lsr->calcLoss(n->lsr, lastL->y, labels);
    }
    return loss;
}
