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
    int nIn;
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
    n->features = VecArrCreate(n->nIn, n->batchSize);
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

Network NetworkCreate(int nIn, Losser lsr, Optimizer o) {
    Network n = CallocOrCrash(sizeof(struct network)); //zero batch size
    n->nIn = nIn;
    n->features = VecArrCreate(n->nIn, n->batchSize);
    n->dFeatures = VecArrCreateSameDim(n->features);
    n->layers = ListInit(sizeof(Layer));
    n->lsr = lsr;
    n->o = o;
    return n;
}

void NetworkAddLayer(Network n, Layer l) {
    int nIn = n->nIn;
    if (n->layers.len > 0) {
        Layer l = ListGetIdx(&n->layers, n->layers.len -1);
        nIn = l->nOut;
    }
    ListAdd(&n->layers, l);
    l->init(l, nIn);
}

//must be called internally after network creation at least once
static void networkSetBatchSize(Network n, int batchSize) {
    if (n->batchSize == batchSize) return;
    networkUnmount(n);
    n->batchSize = batchSize;
    networkMount(n);
}

//assumes proper mounting
static void networkForward(Network n) {
    struct vecArr x = n->features;
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = ListGetIdx(&n->layers, i);
        l->forward(l, x);
        x = l->y;
    }
    n->lsr->forward(n->lsr, x);
}

static void optimizeLayer(Layer l) {
    for (int i = 0; i < l->p.len; i++) {
        struct vecArr p = *(struct vecArr*)ListGetIdx(&l->p, i);
        struct vecArr dp = *(struct vecArr*)ListGetIdx(&l->dp, i);
        Optimizer o = ListGetIdx(&l->optimizers, i);
        o->optimize(o, p, dp);
    }
}

//assumes proper mounting
static void networkBackwardAndOptimize(Network n) {
    Layer prevL = ListGetIdx(&n->layers, n->layers.len -1);
    n->lsr->backward(n->lsr, prevL->y, n->labels, prevL->dy);
    Layer l = prevL;
    for (int i = n->layers.len -1; i >= 1; i--) {
        prevL = ListGetIdx(&n->layers, i -1);
        l->backward(l, prevL->y, prevL->dy);
        optimizeLayer(l);
        prevL = l;
    }
    l->backward(l, n->features, n->dFeatures);
    optimizeLayer(l);
}

static float networkTrainBatch(Network n, Dataset d, int sampleIdxStart) {
        d->getTrainFeatures(d, n->features, sampleIdxStart);
        d->getTrainLabels(d, n->labels, sampleIdxStart);
        networkForward(n);
        Layer lastL = ListGetIdx(&n->layers, n->layers.len -1);
        float loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTrainSamples;
        networkBackwardAndOptimize(n);
        return loss;
}

float NetworkTrain(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    float loss = 0;
    int i = 0;
    for (; i + batchSize < d->nTrainSamples; i += batchSize) loss += networkTrainBatch(n, d, i * batchSize);
    networkSetBatchSize(n, d->nTrainSamples - i * batchSize);
    loss += networkTrainBatch(n, d, i * batchSize);
    return loss;
}

static float networkTestBatch(Network n, Dataset d, int sampleIdxStart) {
        d->getTestFeatures(d, n->features, sampleIdxStart);
        d->getTestLabels(d, n->labels, sampleIdxStart);
        networkForward(n);
        Layer lastL = ListGetIdx(&n->layers, n->layers.len -1);
        float loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTestSamples;
        networkBackwardAndOptimize(n);
        return loss;
}

float NetworkTest(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    float loss = 0;
    int i = 0;
    for (; i + batchSize < d->nTestSamples; i += batchSize) loss += networkTestBatch(n, d, i * batchSize);
    networkSetBatchSize(n, d->nTestSamples - i * batchSize);
    loss += networkTestBatch(n, d, i * batchSize);
    return loss;
}

float NetworkInferTrainSample(Network n, Dataset d, int idx, float** features, float** predictions) {
    if (idx < 0 || idx >= d->nTrainSamples) ErrorAndCrash("invalid sample index chosen");
    networkSetBatchSize(n, 1);
    d->getTrainFeatures(d, n->features, idx);
    d->getTrainLabels(d, n->labels, idx);
    networkForward(n);
    *features = n->features.elems;
    *predictions = n->lsr->y.elems;
    struct vecArr x = n->features;
    if (n->layers.len > 0) {
        Layer l = ListGetIdx(&n->layers, n->layers.len -1);
        x = l->y;
    }
    return n->lsr->calcLoss(n->lsr, x, n->labels);
}

float NetworkInferTestSample(Network n, Dataset d, int idx, float** features, float** predictions) {
    if (idx < 0 || idx >= d->nTestSamples) ErrorAndCrash("invalid sample index chosen");
    networkSetBatchSize(n, 1);
    d->getTestFeatures(d, n->features, idx);
    d->getTestLabels(d, n->labels, idx);
    networkForward(n);
    *features = n->features.elems;
    *predictions = n->lsr->y.elems;
    struct vecArr x = n->features;
    if (n->layers.len > 0) {
        Layer l = ListGetIdx(&n->layers, n->layers.len -1);
        x = l->y;
    }
    return n->lsr->calcLoss(n->lsr, x, n->labels);
}
