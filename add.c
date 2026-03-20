#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "oann.h"
#include "util.h"
#include "layer.h"
//vector addition layer

static void addInit(Layer l, int nIn, OANNfloat varScalingNextLayer) {
    l->nOut = nIn;
    l->p = PtrListInit();
    VecArr b = VecArrCreate(nIn, 1);
    VecArrInitConst(b, 0);
    PtrListAdd(&(l->p), b);
}

static vecArr addForward(Layer l, VecArr x) {
    VecArr b = PtrListGetIdx(l->p, 0);
    for (int i = 0; i < VecArrNVecs(x); i++) {
        for (int j = 0; j < VecArrVecLen(l->y); j++) {
            l->y[j + i * VecArrVecLen(l->y)] = x[j + i * VecArrVecLen(x)] + b[j];
        }
    }
    return l->y;
}

static VecArr addBackward(Layer l, VecArr x, VecArr dx) {
    VecArr b = PtrListGetIdx(l->p, 0);
    VecArr db = PtrListGetIdx(l->dp, 0);
    for (int i = 0; i < VecArrVecLen(b); i++) {
        OANNfloat sum = 0;
        for (int j = 0; j < VecArrNVecs(l->y); j++) {
            sum += l->dy[i + j * VecArrVecLen(l->y)];
        }
    }
    return l->dy;
}

static void addCreate() {
    Layer l = malloc(sizeof(struct Layer));
    l->type = LAYER_ADD;
    l->varScaling = 1;
    l->init = addInit;
    l->forward = addForward;
    l->backward = addBackward;
    return l;
}
