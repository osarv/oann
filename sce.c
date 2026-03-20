#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "losser.h"
//softmax cross entropy loss

static void sceUnmount(Losser l) {
    VecArrDestroy(l->y);
}

static void sceMount(Losser l, VecArr x) {
    l->y = VecArrCreateSameDim(x);
}

static void sceDestroy(Losser l) {
    free(l);
}

static void sceForward(Losser l, VecArr x) {
    for (int i = 0; i < VecArrNVecs(x); i++) {
        OANNfloat maxVal = 0;
        for (int j = 0; j < VecArrVecLen(x); j++) {
            if (x[i * VecArrVecLen(x) + j] > maxVal) maxVal = x[i * VecArrVecLen(x) + j];
        }
        OANNfloat sum = 0;
        for (int j = 0; j < VecArrVecLen(x); j++) {
            l->y[i * VecArrVecLen(x) + j] = exp(x[i * VecArrVecLen(x) + j] - maxVal);
            sum += l->y[i * VecArrVecLen(x) + j];
        }
        for (int j = 0; j < VecArrVecLen(x); j++) l->y[i * VecArrVecLen(x) + j] /= sum;
    }
}

static void sceBackward(Losser l, VecArr x, VecArr labels, VecArr dx) {
    (void)x; //correct implementation
    for (int i = 0; i < VecArrNElems(l->y); i++) dx[i] = l->y[i] - labels[i];
}

#define SCE_EPSILON 1e-10f
static OANNfloat sceCalcLoss(Losser l, VecArr x, VecArr labels) {
    (void)x; //correct implementation
    OANNfloat loss = 0;
    for (int i = 0; i < VecArrNElems(l->y); i++) loss -= labels[i] * log(l->y[i] + SCE_EPSILON);
    return loss;
}

TEST(SCE) {
    Losser l = SceCreate();
    VecArr x = VecArrCreate(3, 2);
    VecArr dx = VecArrCreateSameDim(x);
    VecArr labels = VecArrCreateSameDim(x);
    l->mount(l, x);
    for (int i = 0; i < 6; i++) {
        x[i] = i;
        labels[i] = 0;
    }
    labels[1] = 1;
    labels[5] = 1;

    OANNfloat yDesired[6];
    OANNfloat dxDesired[6];
    OANNfloat sum1 = exp(0) + exp(1) + exp(2);
    OANNfloat sum2 = exp(3) + exp(4) + exp(5);
    OANNfloat lossDesired = 0;
    for (int i = 0; i < 3; i++) yDesired[i] = exp(i) / sum1;
    for (int i = 3; i < 6; i++) yDesired[i] = exp(i) / sum2;
    for (int i = 0; i < 6; i++) {
        dxDesired[i] = yDesired[i] - labels[i];
        lossDesired -= labels[i] * log(yDesired[i]);
    }
    l->forward(l, x);
    l->backward(l, x, labels, dx);
    OANNfloat loss = l->calcLoss(l, x, labels);
    if (!VecArrOANNfloatArrIsEqual(l->y, yDesired)) TEST_FAILED
    if (!VecArrOANNfloatArrIsEqual(dx, dxDesired)) TEST_FAILED
    if (loss != lossDesired) TEST_FAILED
    l->unmount(l);
    l->destroy(l);
    TEST_PASSED
}

Losser SceCreate() {
    struct losser* s = MallocOrCrash(sizeof(struct losser));
    s->unmount = sceUnmount;
    s->mount = sceMount;
    s->destroy = sceDestroy;
    s->forward = sceForward;
    s->backward = sceBackward;
    s->calcLoss = sceCalcLoss;
    return s;
}
