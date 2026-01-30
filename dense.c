#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include "layer.h"
#include "operation.h"
//fully connected layer

struct dense {
    struct layer l;
    VecArr w;
    VecArr dw;
    VecArr b;
    VecArr db;
};

static void denseUnmount(Layer l) {
    VecArrDestroy(l->y);
    VecArrDestroy(l->dy);
}

static void denseMount(Layer l, VecArr x) {
    l->y = VecArrCreate(l->nOut, VecArrNVecs(x));
    l->dy = VecArrCreate(l->nOut, VecArrNVecs(x));
}

static void denseInit(Layer l, int nIn, float varScalingNextLayer) {
    struct dense* d = (struct dense*)l;
    d->w = VecArrCreate(nIn, l->nOut);
    d->dw = VecArrCreateSameDim(d->w);
    d->b = VecArrCreate(l->nOut, 1);
    d->db = VecArrCreateSameDim(d->b);
    l->p = PtrListInit();
    l->dp = PtrListInit();
    PtrListAdd(&l->p, d->w);
    PtrListAdd(&l->p, d->b);
    PtrListAdd(&l->dp, d->dw);
    PtrListAdd(&l->dp, d->db);
    VecArrInitNormDist(d->w, 0, sqrt(1 / (nIn * varScalingNextLayer)));
    VecArrInitConst(d->b, 0);
}

static void denseDestroy(Layer l) {
    struct dense* d = (struct dense*)l;
    VecArrDestroy(d->w);
    VecArrDestroy(d->b);
    PtrListDestroy(l->p);
    PtrListDestroy(l->dp);
    PtrListDestroy(l->optimizers);
    free(d);
}

#ifdef OP_MODE_BLAS
static void denseForward(Layer l, VecArr x) {
    struct dense* d = (struct dense*)l;

    int nIn = VecArrVecLen(x);
    int nOut = l->nOut;
    int batchSize = VecArrNVecs(x);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            batchSize, nOut, nIn, 1.0f, x, nIn,
            d->w, nIn, 0.0f, l->y, nOut);

    for (int i = 0; i < batchSize; i++) {
        cblas_saxpy(nOut, 1.0f, d->b, 1, l->y + i * nOut, 1);
    }
}

static void denseBackward(Layer l, VecArr x, VecArr dx) {
    struct dense* d = (struct dense*)l;

    int nIn = VecArrVecLen(x);
    int nOut = l->nOut;
    int batchSize = VecArrNVecs(x);

    //calculate gradients downstream
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            batchSize, nIn, nOut, 1.0f, l->dy, nOut,
            d->w, nIn, 0.0f, dx, nIn);

    //calculate weight gradients
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            nOut, nIn, batchSize, 1.0f, l->dy, nOut,
            x, nIn, 0.0f, d->dw, nIn);

    //calculate bias gradients
    VecArrInitConst(d->db, 0);
    for (int i = 0; i < batchSize; i++) cblas_saxpy(nOut, 1.0f, l->dy + i * nOut, 1, d->db, 1);
}
#endif //OP_MODE_BLAS

Layer DenseCreate(int nOut) {
    struct dense* d = CallocOrCrash(sizeof(struct dense)); //calloc needed to initialized lists to zero length
    d->l.varScaling = 1;
    d->l.nOut = nOut;
    d->l.unmount = denseUnmount;
    d->l.mount = denseMount;
    d->l.init = denseInit;
    d->l.destroy = denseDestroy;
    d->l.forward = denseForward;
    d->l.backward = denseBackward;
    return &d->l;
}

TEST(Dense) {
    Layer l = DenseCreate(2);
    struct dense* d = (struct dense*)l;
    VecArr x = VecArrCreate(3, 2);
    VecArr dx = VecArrCreateSameDim(x);
    l->init(l, 3, 1);
    l->mount(l, x);
    for (int i = 0; i < 6; i++) {
        d->w[i] = i;
        x[i] = i;
    }
    for (int i = 0; i < 4; i++) l->dy[i] = i;
    for (int i = 0; i < 2; i++) d->b[i] = i;
    
    float yDesired[] = {5, 15, 14, 51};
    float dwDesired[] = {6, 8, 10, 9, 13, 17};
    float dbDesired[] = {2, 4};
    float dxDesired[] = {3, 4, 5, 9, 14, 19};

    l->forward(l, x);
    if (!VecArrFloatArrIsEqual(l->y, yDesired)) TEST_FAILED
    l->backward(l, x, dx);
    if (!VecArrFloatArrIsEqual(d->dw, dwDesired)) TEST_FAILED
    if (!VecArrFloatArrIsEqual(d->db, dbDesired)) TEST_FAILED
    if (!VecArrFloatArrIsEqual(dx, dxDesired)) TEST_FAILED
    if (l->p.len != 2) TEST_FAILED;
    if (l->dp.len != 2) TEST_FAILED;
    if (PtrListGetIdx(l->p, 0) != d->w) TEST_FAILED;
    if (PtrListGetIdx(l->p, 1) != d->b) TEST_FAILED;
    if (PtrListGetIdx(l->dp, 0) != d->dw) TEST_FAILED;
    if (PtrListGetIdx(l->dp, 1) != d->db) TEST_FAILED;
    l->unmount(l);
    l->destroy(l);
    TEST_PASSED
}
