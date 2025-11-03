#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include "layer.h"
#include "util.h"

struct layer {
    enum layerType type;
    int nOut;
    int nVectors;
    int nBuckets;
    int rehashRate;
    Matrix hashVectors;
    katrix preHashes;
    Matrix passHashes;
    Matrix weights;
    Matrix biases;
    Matrix outF;
    Matrix outB;
    void (*init)(Layer l, Matrix inF);
    Matrix (*forward)(Layer l, Matrix inF);
    Matrix (*backward)(Layer l, Matrix inB);
};

void LayerDestroy(Layer l) {
    free(l);
}

void LayerMount(Layer l, Matrix inF) {
    l->outF = MatrixCreate(l->nOut, inF->nCols);
    l->outB = MatrixCreate(inF->nRows, inF->nCols);
    l->passHashes = MatrixCreate();
}

static void hashWeights(Layer l) {
    MatrixMul(l->weights, l->hashVectors, l->hashes, false, true);
}

void HashDenseInit(Layer l, Matrix inF) {
    int nIn = inF->nRows;
    l->weights = MatrixCreate(l->nOut, nIn);
    l->biases = MatrixCreate(l->nOut, 1);
    l->hashVectors = MatrixCreate(l->nVectors, nIn);
    l->preHashes = MatrixCreate(l->nOut, l->nVectors);
    MatrixInitNormDist(l->weights, 0, sqrt(2.0 / (float)inF.nRows));
    MatrixInitConst(l->biases, 0);
    MatrixInitNormDist(l->hashVectors, 0, 1);
    hashWeights(l);
}

static void hashInput(Layer l, Matrix inF) {
    MatrixMul(l->hashVectors, inF, false, true, );
}

Matrix HashDenseForward(Layer l, Matrix inF) {
    MatrixMul();
}

Matrix HashDenseBackward() {

}

Layer HashDenseCreate(int nOut, int nVectors, int nBuckets, int rehashRate) {
    Layer l = MallocOrCrash(sizeof(struct layer));

    l->type = LAYER_HASHDENSE;
    l->nOut = nOut;
    l->nVectors = nVectors;
    l->nBuckets = nBuckets;
    l->rehashRate = rehashRate;
    return l;
}
