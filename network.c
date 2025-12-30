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
    VecArrDestroy(n->features);
    VecArrDestroy(n->dFeatures);
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = PtrListGetIdx(n->layers, i);
        l->unmount(l);
    }
    n->lsr->unmount(n->lsr);
    VecArrDestroy(n->labels);
}

static void networkMount(Network n) {
    n->features = VecArrCreate(n->nIn, n->batchSize);
    n->dFeatures = VecArrCreateSameDim(n->features);
    VecArr x = n->features;
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = PtrListGetIdx(n->layers, i);
        l->mount(l, x);
        x = l->y;
    }
    n->lsr->mount(n->lsr, x);
    n->labels = VecArrCreate(VecArrVecLen(n->lsr->y), n->batchSize);
}

void NetworkDestroy(Network n) {
    if (n->batchSize != 0) networkUnmount(n);
    for (int i = 0; i < n->layers.len; i++) {
        Layer l = PtrListGetIdx(n->layers, i);
        l->destroy(l);
    }
    n->lsr->destroy(n->lsr);
    n->oRecipe->destroy(n->oRecipe);
    free(n);
}

Network NetworkCreate(int nIn, Losser lsr, Optimizer o) {
    Network n = CallocOrCrash(sizeof(struct network)); //set batchSize to zero
    n->nIn = nIn;
    n->features = VecArrCreate(n->nIn, n->batchSize);
    n->dFeatures = VecArrCreateSameDim(n->features);
    n->layers = PtrListInit();
    n->lsr = lsr;
    n->oRecipe = o;
    return n;
}

void NetworkAddLayer(Network n, Layer l) {
    int nIn = n->nIn;
    if (n->layers.len > 0) {
        Layer l = PtrListGetIdx(n->layers, n->layers.len -1);
        nIn = l->nOut;
    }
    l->init(l, nIn);
    for (int i = 0; i < l->p.len; i++) {
        VecArr p = PtrListGetIdx(l->p, i);
        Optimizer o = n->oRecipe->yieldOptimizer(n->oRecipe, p);
        PtrListAdd(&l->optimizers, o);
    }
    PtrListAdd(&n->layers, l);
}

//must be called internally after network creation at least once
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
        prevL = l;
    }
    l->backward(l, n->features, n->dFeatures);
    optimizeLayer(l);
}

static float networkTrainBatch(Network n, Dataset d, int sampleIdxStart) {
    d->getTrainFeatures(n->features, sampleIdxStart);
    d->getTrainLabels(n->labels, sampleIdxStart);
    networkForward(n);
    Layer lastL = PtrListGetIdx(n->layers, n->layers.len -1);
    float loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTrainSamples;
    networkBackwardAndOptimize(n);
    return loss;
}

float NetworkTrain(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    float loss = 0;
    int i = 0;
    for (; i + batchSize < d->nTrainSamples; i += batchSize) loss += networkTrainBatch(n, d, i);
    networkSetBatchSize(n, d->nTrainSamples - i * batchSize);
    loss += networkTrainBatch(n, d, i * batchSize);
    return loss;
}

static float networkTestBatch(Network n, Dataset d, int sampleIdxStart) {
        d->getTestFeatures(n->features, sampleIdxStart);
        d->getTestLabels(n->labels, sampleIdxStart);
        networkForward(n);
        Layer lastL = PtrListGetIdx(n->layers, n->layers.len -1);
        float loss = n->lsr->calcLoss(n->lsr, lastL->y, n->labels) / d->nTestSamples;
        networkBackwardAndOptimize(n);
        return loss;
}

float NetworkTest(Network n, int batchSize, Dataset d) {
    networkSetBatchSize(n, batchSize);
    float loss = 0;
    int i = 0;
    for (; i + batchSize < d->nTestSamples; i += batchSize) loss += networkTestBatch(n, d, i);
    networkSetBatchSize(n, d->nTestSamples - i * batchSize);
    loss += networkTestBatch(n, d, i * batchSize);
    return loss;
}

float NetworkInferTrainSample(Network n, Dataset d, int idx, float** features, float** predictions) {
    if (idx < 0 || idx >= d->nTrainSamples) ErrorAndCrash("invalid sample index chosen");
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

float NetworkInferTestSample(Network n, Dataset d, int idx, float** features, float** predictions) {
    if (idx < 0 || idx >= d->nTestSamples) ErrorAndCrash("invalid sample index chosen");
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
