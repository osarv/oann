#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
//rectified linear unit layer

static void reluInit(Layer l, int nIn, OANNfloat varScalingNextLayer) {
    l->nOut = nIn;
    (void)varScalingNextLayer; //correct implementation
}

static void reluForward(Layer l, VecArr x) {
    for (int i = 0; i < VecArrNElems(x); i++) {
        l->y[i] = x[i] > 0 ? x[i] : 0;
    }
}

static void reluBackward(Layer l, VecArr x, VecArr dx) {
    for (int i = 0; i < VecArrNElems(x); i++) {
        dx[i] = x[i] > 0 ? l->dy[i] : 0;
    }
}

Layer ReluCreate() {
    Layer l = MallocOrCrash(sizeof(struct layer));
    l->varScaling = 0.5;
    l->init = reluInit;
    l->forward = reluForward;
    l->backward = reluBackward;
    return l;
}

TEST(Relu) {
    Layer l = ReluCreate();
    VecArr x = VecArrCreate(3, 2);
    VecArr dx = VecArrCreateSameDim(x);
    l->y = VecArrCreateSameDim(x);
    l->dy = VecArrCreateSameDim(x);
    l->init(l, 3, 1);
    for (int i = 0; i < 6; i++) {
        x[i] = i -2;
        l->dy[i] = i;
    }
    OANNfloat yDesired[] = {0, 0, 0, 1, 2, 3};
    OANNfloat dxDesired[] = {0, 0, 0, 3, 4, 5};
    l->forward(l, x);
    if (!VecArrOANNfloatArrIsEqual(l->y, yDesired)) TEST_FAILED
    l->backward(l, x, dx);
    if (!VecArrOANNfloatArrIsEqual(dx, dxDesired)) TEST_FAILED
    TEST_PASSED
}
