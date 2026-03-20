#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oann.h"
#include "util.h"
#include "layer.h"
//matrix multiplication layer

static void mulInit(Layer l, int nIn, OANNfloat varScalingNextLayer) {
    l->p = PtrListInit();
    VecArr w = VecArrCreate(nIn, l->nOut);
    VecArrInitNormDist(w, 0, sqrt(1 / (nIn * varScalingNextLayer)));
    PtrListAdd(&(l->p), w);
}

static VecArr mulForward(Layer l, VecArr x) {
    VecArr w = PtrListGetIdx(l->p, 0);

    for (int i = 0; i < VecArrNVecs(x); i++) {
        for (int j = 0; j < VecArrVecLen(l->y); j++) {
            OANNfloat sum = 0;
            for (int k = 0; k < VecArrVecLen(x); k++) {
                sum += x[k + i * VecArrVecLen(x)] * w[k + j * VecArrVecLen(x)];
            }
            l->y[j + i * VecArrVecLen(l->y)] = sum;
        }
    }
    return l->y;
}

static VecArr mulBackward(Layer l, VecArr x, VecArr dx) {
    VecArr w = PtrListGetIdx(l->p, 0);
    VecArr dw = PtrListGetIdx(l->dp, 0);
    for (int i = 0; i < VecArrNVecs(x); i++) {
        for (int j = 0; j < VecArrVecLen(x); j++) {
            OANNfloat sum = 0;
            for (int k = 0; k < VecArrNVecs(w); k++) {
                sum += l->dy[k + i * VecArrVecLen(l->y)] * w[j + k * VecArrVecLen(w)];
            }
            dx[j + i * VecArrVecLen(x)] = sum;
        }
    }

    for (int i = 0; i < VecArrNVecs(w); i++) {
        for (int j = 0; j < VecArrVecLen(w); j++) {
            OANNfloat sum = 0;
            for (int k = 0; k < VecArrNVecs(x); k++) {
                sum += x[j + k * VecArrVecLen(x)] * l->dy[i + k * VecArrVecLen(l->y)];
            }
            dw[j + i * VecArrVecLen(x)] = sum;
        }
    }
    return dx;
}

Layer MulCreate(int nOut) {
    Layer l = MallocOrCrash(sizeof(struct layer));
    l->type = LAYER_MUL;
    l->varScaling = 1;
    l->nOut = nOut;
    l->init = mulInit;
    l->forward = mulForward;
    l->backward = mulBackward;
    return l;
}

TEST(Mul) {
    Layer l = MulCreate(2);
    VecArr x = VecArrCreate(3, 2);
    VecArr dx = VecArrCreateSameDim(x);
    l->init(l, VecArrVecLen(x), 1);
    l->y = VecArrCreate(2, 2);
    l->dy = VecArrCreateSameDim(l->y);
    VecArr w = PtrListGetIdx(l->p, 0);
    VecArr dw = VecArrCreateSameDim(w);
    l->dp = PtrListInit();
    PtrListAdd(&(l->dp), dw);
    for (int i = 0; i < VecArrNElems(x); i++) x[i] = i + 1;
    for (int i = 0; i < VecArrNElems(w); i++) w[i] = i + 3;
    for (int i = 0; i < VecArrNElems(l->dy); i++) l->dy[i] = i;
    OANNfloat yExpected[] = {26, 44, 62, 107};
    OANNfloat dxExpected[] = {6, 7, 8, 24, 29, 34};
    OANNfloat dwExpected[] = {8, 10, 12, 13, 17, 21};
    l->forward(l, x);
    l->backward(l, x, dx);
    if (!VecArrOANNfloatArrIsEqual(l->y, yExpected)) TEST_FAILED
    if (!VecArrOANNfloatArrIsEqual(dx, dxExpected)) TEST_FAILED
    if (!VecArrOANNfloatArrIsEqual(dw, dwExpected)) TEST_FAILED
    TEST_PASSED
}

