#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <cblas.h>
#include "operation.h"
#include "util.h"

void OperationReluForward(struct vecArr x, struct vecArr y) {
    for (int i = 0; i < x.vecLen * x.nVecs; i++) {
        y.elems[i] = x.elems[i] > 0.0f? x.elems[i] : 0.0f;
    }
}

void OperationReluBackward(struct vecArr x, struct vecArr dy, struct vecArr dx) {
    for (int i = 0; i < x.vecLen * x.nVecs; i++) {
        dx.elems[i] = x.elems[i] > 0.0f? dy.elems[i] : 0.0f;
    }
}

#ifdef OP_MODE_BLAS
void OperationDenseForward(struct vecArr x, struct vecArr w, struct vecArr b, struct vecArr y) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            w.nVecs, x.nVecs, w.vecLen, 1.0f, w.elems, w.vecLen,
            x.elems, x.vecLen, 0.0f, y.elems, y.vecLen);

    for (int i = 0; i < y.nVecs; i++) {
        cblas_saxpy(b.vecLen, 1.0f, b.elems, 1, y.elems + i * y.vecLen, 1);
    }
}

void OperationDenseBackward(struct vecArr x, struct vecArr dy, struct vecArr w, struct vecArr dw, struct vecArr dx) {

    //calculate weight gradients
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            w.nVecs, w.vecLen, x.nVecs, 1.0f, dy.elems, dy.vecLen,
            x.elems, x.vecLen, 0.0f, dw.elems, dw.vecLen);

    //calculate gradients downstream
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            x.nVecs, w.vecLen, w.nVecs, 1.0f, dy.elems, dy.vecLen,
            w.elems, w.vecLen, 0.0f, dx.elems, dx.vecLen);
}
#endif //OP_MODE_BLAS

TEST(OperationDenseForwardAndBackward) {
    struct vecArr x = VecArrCreate(3, 2);
    struct vecArr dy = VecArrCreate(2, 2);
    struct vecArr y = VecArrCreate(2, 2);
    struct vecArr yDesired = VecArrCreate(2, 2);
    struct vecArr dx = VecArrCreate(3, 2);
    struct vecArr dxDesired = VecArrCreate(3, 2);
    struct vecArr w = VecArrCreate(3, 2);
    struct vecArr b = VecArrCreate(2, 1);
    struct vecArr dw = VecArrCreate(3, 2);
    struct vecArr dwDesired = VecArrCreate(3, 2);
    for (int i = 0; i < 6; i++) {x.elems[i] = i; w.elems[i] = i; dw.elems[i] = i; dx.elems[i] = i;}
    for (int i = 0; i < 4; i++) {y.elems[i] = i; dy.elems[i] = i;}
    for (int i = 0; i < 2; i++) b.elems[i] = i;
    
    float yDesiredElems[] = {
        5, 15,
        14, 51
    };

    float dxDesiredElems[] = {
        6, 8, 10,
        9, 13, 17
    };

    float dwDesiredElems[] = {
        3, 4, 5,
        9, 14, 19
    };

    yDesired.elems = yDesiredElems;
    dxDesired.elems = dxDesiredElems;
    dwDesired.elems = dwDesiredElems;

    OperationDenseForward(x, w, b, y);
    if (!VecArrIsEqual(y, yDesired)) TEST_FAILED
    OperationDenseBackward(x, dy, w, dw, dx);
    if (!VecArrIsEqual(dw, dwDesired)) TEST_FAILED
    if (!VecArrIsEqual(dx, dxDesired)) TEST_FAILED
    TEST_PASSED
}

TEST(OperationReluForwardAndBackward) {
    struct vecArr x = VecArrCreate(3, 2);
    struct vecArr dy = VecArrCreate(3, 2);
    struct vecArr y = VecArrCreate(3, 2);
    struct vecArr yDesired = VecArrCreate(3, 2);
    struct vecArr dx = VecArrCreate(3, 2);
    struct vecArr dxDesired = VecArrCreate(3, 2);
    for (int i = 0; i < 6; i++) {
        x.elems[i] = i -2;
        dy.elems[i] = i;
        y.elems[i] = i -2;
        dx.elems[i] = i -2;
    }

    float yDesiredElems[] = {0, 0, 0, 1, 2, 3};
    float dxDesiredElems[] = {0, 0, 0, 3, 4, 5};
    yDesired.elems = yDesiredElems;
    dxDesired.elems = dxDesiredElems;
    OperationReluForward(x, y);
    if (!VecArrIsEqual(y, yDesired)) TEST_FAILED
    OperationReluBackward(x, dy, dx);
    if (!VecArrIsEqual(dx, dxDesired)) TEST_FAILED
    TEST_PASSED
}

void OperationSCEForward(struct vecArr x, struct vecArr y) {
    for (int i = 0; i < x.nVecs; i++) {
        float maxVal = 0;
        for (int j = 0; j < x.vecLen; j++) {
            if (x.elems[i * x.vecLen + j] > maxVal) maxVal = x.elems[i * x.vecLen + j];
        }
        float sum = 0;
        for (int j = 0; j < x.vecLen; j++) {
            y.elems[i * x.vecLen + j] = exp(x.elems[i * x.vecLen + j] - maxVal);
            sum += y.elems[i * x.vecLen + j];
        }
        for (int j = 0; j < x.vecLen; j++) y.elems[i * x.vecLen + j] /= sum;
    }
}

void OperationSCEBackward(struct vecArr y, struct vecArr labels, struct vecArr dx) {
    for (int i = 0; i < y.nVecs * y.vecLen; i++) dx.elems[i] = y.elems[i] - labels.elems[i];
}

float OperationSCECalcLoss(struct vecArr y, struct vecArr labels) {
    float loss = 0;
    for (int i = 0; i < y.nVecs * y.vecLen; i++) loss -= labels.elems[i] * log(y.elems[i]);
    return loss;
}

TEST(OperationSCEForwardAndBackward) {
    struct vecArr x = VecArrCreate(3, 2);
    struct vecArr y = VecArrCreate(3, 2);
    struct vecArr dx = VecArrCreate(3, 2);
    struct vecArr labels = VecArrCreate(3, 2);
    struct vecArr yDesired = VecArrCreate(3, 2);
    struct vecArr dxDesired = VecArrCreate(3, 2);
    for (int i = 0; i < 6; i++) {
        x.elems[i] = i;
        y.elems[i] = i;
        dx.elems[i] = i;
        labels.elems[i] = 0;
    }
    labels.elems[1] = 1;
    labels.elems[5] = 1;
    float yDesiredElems[6];
    float dxDesiredElems[6];
    float sum1 = exp(0) + exp(1) + exp(2);
    float sum2 = exp(3) + exp(4) + exp(5);
    for (int i = 0; i < 3; i++) yDesiredElems[i] = exp(i) / sum1;
    for (int i = 3; i < 6; i++) yDesiredElems[i] = exp(i) / sum2;
    for (int i = 0; i < 6; i++) dxDesiredElems[i] = yDesiredElems[i] - labels.elems[i];
    yDesired.elems = yDesiredElems;
    dxDesired.elems = dxDesiredElems;
    OperationSCEForward(x, y);
    OperationSCEBackward(y, labels, dx);
    if (!VecArrIsEqual(y, yDesired)) TEST_FAILED
    if (!VecArrIsEqual(dx, dxDesired)) TEST_FAILED
    TEST_PASSED
}
