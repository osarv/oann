#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
#include "operation.h"
//rectified linear unit layer

static void reluUnmount(Layer l) {
    VecArrDestroy(l->y);
    VecArrDestroy(l->dy);
}

static void reluMount(Layer l, VecArr x) {
    l->y = VecArrCreateSameDim(x);
    l->dy = VecArrCreateSameDim(x);
}

static void reluInit(Layer l, int nIn, float varScalingNextLayer) {
    (void)varScalingNextLayer; //correct implementation
    (void)l; //correct implementation
    (void)nIn; //correct implementation
}

static void reluDestroy(Layer l) {
    free(l);
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
    Layer l = CallocOrCrash(sizeof(struct layer)); //calloc needed to initialize lists to zero length
    l->varScaling = 0.5;
    l->unmount = reluUnmount;
    l->mount = reluMount;
    l->init = reluInit;
    l->destroy = reluDestroy;
    l->forward = reluForward;
    l->backward = reluBackward;
    return l;
}

TEST(Relu) {
    Layer l = ReluCreate();
    VecArr x = VecArrCreate(3, 2);
    VecArr dx = VecArrCreateSameDim(x);
    l->init(l, 3, 1);
    l->mount(l, x);
    for (int i = 0; i < 6; i++) {
        x[i] = i -2;
        l->dy[i] = i;
    }
    float yDesired[] = {0, 0, 0, 1, 2, 3};
    float dxDesired[] = {0, 0, 0, 3, 4, 5};
    l->forward(l, x);
    if (!VecArrFloatArrIsEqual(l->y, yDesired)) TEST_FAILED
    l->backward(l, x, dx);
    if (!VecArrFloatArrIsEqual(dx, dxDesired)) TEST_FAILED
    l->unmount(l);
    l->destroy(l);
    TEST_PASSED
}
