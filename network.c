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
    VecArr features;
    VecArr dFeatures; //needed for generic layer backward calling
    VecArr labels;
    struct ptrList layers;
    Losser lsr;
    Optimizer oRecipe;
};

static void networkUnmount(Network n) {
    (void)n;
    //TODO
}

static void networkMount(Network n) {
    (void)n;
    //TODO
}

void NetworkDestroy(Network n) {
    (void)n;
    //TODO
}

Network NetworkCreate(int nIn, Optimizer o) {
    Network n = CallocOrCrash(sizeof(struct network)); //set batchSize to zero
    n->nIn = nIn;
    n->features = VecArrCreate(n->nIn, n->batchSize);
    n->dFeatures = VecArrCreateSameDim(n->features);
    n->layers = PtrListInit();
    n->oRecipe = o;
    return n;
}

void NetworkAddLayer(Network n, Layer l) {
    PtrListAdd(&n->layers, l);
}

void NetworkSetLosser(Network n, Losser lsr) { //completes the network creation
    n->lsr = lsr;

    int nIn = n->nIn;
    Layer l = PtrListGetIdx(n->layers, 0);
    for (int i = 0; i < n->layers.len -1; i++) {
        Layer nextL = PtrListGetIdx(n->layers, i +1);
        l->init(l, nIn, nextL->varScaling);
        for (int j = 0; j < l->p.len; j++) {
            VecArr p = PtrListGetIdx(l->p, j);
            Optimizer o = n->oRecipe->yieldOptimizer(n->oRecipe, p);
            PtrListAdd(&l->optimizers, o);
        }
        nIn = l->nOut;
        l = nextL;
    }

    l->init(l, nIn, 1);
    for (int i = 0; i < l->p.len; i++) {
        VecArr p = PtrListGetIdx(l->p, i);
        Optimizer o = n->oRecipe->yieldOptimizer(n->oRecipe, p);
        PtrListAdd(&l->optimizers, o);
    }
}

//must be called internally before network usage at least once
static void networkSetBatchSize(Network n, int batchSize) {
    if (n->batchSize == batchSize) return;
    if (n->batchSize != 0) networkUnmount(n);
    n->batchSize = batchSize;
    networkMount(n);
}

//assumes proper mounting
static void networkForward(Network n) {
    VecArr x = n->features;
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = PtrListGetIdx(n->layers, i);
        l->forward(l, x);
        x = l->y;
    }
    n->lsr->forward(n->lsr, x);
}

static void optimizeLayer(Layer l) {
    for (int i = 0; i < l->p.len; i++) {
        VecArr p = PtrListGetIdx(l->p, i);
        VecArr dp = PtrListGetIdx(l->dp, i);
        Optimizer o = PtrListGetIdx(l->optimizers, i);
        o->optimize(o, p, dp);
    }
}

//assumes proper mounting
static void networkBackwardAndOptimize(Network n) {
    Layer prevL = PtrListGetIdx(n->layers, n->layers.len -1);
    n->lsr->backward(n->lsr, prevL->y, n->labels, prevL->dy);
    Layer l = prevL;
    for (int i = n->layers.len -1; i >= 1; i--) {
        prevL = PtrListGetIdx(n->layers, i -1);
        l->backward(l, prevL->y, prevL->dy);
        optimizeLayer(l);
        l = prevL;
    }
    l->backward(l, n->features, n->dFeatures);
    optimizeLayer(l);
}

static OANNfloat networkTrainBatch(Network n, Dataset d, int sIdx) {
    d->getTrainFeatures(n->features, sIdx);
    d->getTrainLabels(n->labels, sIdx);
    networkForward(n);
    Layer lastL = PtrListGetIdx(n->layers, n->layers.len -1);
    OANNfloat loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTrainSamples;
    networkBackwardAndOptimize(n);
    return loss;
}

OANNfloat NetworkTrain(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    OANNfloat loss = 0;
    int i = 0;
    for (; i + batchSize <= d->nTrainSamples; i += batchSize) loss += networkTrainBatch(n, d, i);
    if (i < d->nTestSamples) {
        networkSetBatchSize(n, d->nTrainSamples - i);
        loss += networkTrainBatch(n, d, i);
    }
    return loss;
}

OANNfloat NetworkTrainOneBatch(Network n, int batchSize, int sIdx, Dataset d) {
    if (sIdx < 0 || sIdx + batchSize >= d->nTrainSamples) ErrorAndCrash("invalid batch range");
    networkSetBatchSize(n, batchSize);
    return networkTrainBatch(n, d, sIdx);
}

static OANNfloat networkTestBatch(Network n, Dataset d, int sIdx) {
        d->getTestFeatures(n->features, sIdx);
        d->getTestLabels(n->labels, sIdx);
        networkForward(n);
        Layer lastL = PtrListGetIdx(n->layers, n->layers.len -1);
        OANNfloat loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTestSamples;
        networkBackwardAndOptimize(n);
        return loss;
}

OANNfloat NetworkTest(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    OANNfloat loss = 0;
    int i = 0;
    for (; i + batchSize <= d->nTestSamples; i += batchSize) loss += networkTestBatch(n, d, i);
    if (i < d->nTestSamples) {
        networkSetBatchSize(n, d->nTestSamples - i);
        loss += networkTestBatch(n, d, i);
    }
    return loss;
}

OANNfloat NetworkTestOneBatch(Network n, int batchSize, int sIdx, Dataset d) {
    if (sIdx < 0 || sIdx + batchSize >= d->nTestSamples) ErrorAndCrash("invalid batch range");
    networkSetBatchSize(n, batchSize);
    return networkTestBatch(n, d, sIdx);
}

OANNfloat NetworkInferTrainSample(Network n, Dataset d, int idx, OANNfloat** features, OANNfloat** predictions) {
    if (idx < 0 || idx >= d->nTrainSamples) ErrorAndCrash("invalid sample index");
    networkSetBatchSize(n, 1);
    d->getTrainFeatures(n->features, idx);
    d->getTrainLabels(n->labels, idx);
    networkForward(n);
    *features = n->features;
    *predictions = n->lsr->y;
    VecArr x = n->features;
    if (n->layers.len > 0) {
        Layer l = PtrListGetIdx(n->layers, n->layers.len -1);
        x = l->y;
    }
    return n->lsr->calcLoss(n->lsr, x, n->labels);
}

OANNfloat NetworkInferTestSample(Network n, Dataset d, int idx, OANNfloat** features, OANNfloat** predictions) {
    if (idx < 0 || idx >= d->nTestSamples) ErrorAndCrash("invalid sample index");
    networkSetBatchSize(n, 1);
    d->getTestFeatures(n->features, idx);
    d->getTestLabels(n->labels, idx);
    networkForward(n);
    *features = n->features;
    *predictions = n->lsr->y;
    VecArr x = n->features;
    if (n->layers.len > 0) {
        Layer l = PtrListGetIdx(n->layers, n->layers.len -1);
        x = l->y;
    }
    return n->lsr->calcLoss(n->lsr, x, n->labels);
}
