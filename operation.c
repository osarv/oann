#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cblas.h>
#include "operation.h"
#include "util.h"

void OperationReluForward(struct vecArr inF, struct vecArr outF) {
    for (int i = 0; i < inF.vecLen * inF.nVecs; i++) {
        outF.elems[i] = inF.elems[i] > 0.0f? inF.elems[i] : 0.0f;
    }
}

void OperationReluBackward(struct vecArr inF, struct vecArr inB, struct vecArr outB) {
    for (int i = 0; i < inF.vecLen * inF.nVecs; i++) {
        outB.elems[i] = inF.elems[i] > 0.0f? inB.elems[i] : 0.0f;
    }
}

#ifdef OP_MODE_BLAS
void OperationDenseForward(struct vecArr inF, struct vecArr w, struct vecArr b, struct vecArr outF) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            w.nVecs, inF.nVecs, w.vecLen, 1.0f, w.elems, w.vecLen,
            inF.elems, inF.vecLen, 0.0f, outF.elems, outF.vecLen);

    for (int i = 0; i < outF.nVecs; i++) {
        cblas_saxpy(b.vecLen, 1.0f, b.elems, 1, outF.elems + i * outF.vecLen, 1);
    }
}

void OperationDenseBackward(struct vecArr inF, struct vecArr inB, struct vecArr w, struct vecArr wGrads, struct vecArr outB) {

    //calculate weight gradients
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            w.nVecs, w.vecLen, inF.nVecs, 1.0f, inB.elems, inB.vecLen,
            inF.elems, inF.vecLen, 0.0f, wGrads.elems, wGrads.vecLen);

    //calculate gradient downstream
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            inF.nVecs, w.vecLen, w.nVecs, 1.0f, inB.elems, inB.vecLen,
            w.elems, w.vecLen, 0.0f, outB.elems, outB.vecLen);
}
#endif //OP_MODE_BLAS

TEST(OperationDenseForwardAndBackward)() {
    struct vecArr inF = VecArrCreate(3, 2);
    struct vecArr inB = VecArrCreate(2, 2);
    struct vecArr outF = VecArrCreate(2, 2);
    struct vecArr outFDesired = VecArrCreate(2, 2);
    struct vecArr outB = VecArrCreate(3, 2);
    struct vecArr outBDesired = VecArrCreate(3, 2);
    struct vecArr w = VecArrCreate(3, 2);
    struct vecArr b = VecArrCreate(2, 1);
    struct vecArr wGrads = VecArrCreate(3, 2);
    struct vecArr wGradsDesired = VecArrCreate(3, 2);
    for (int i = 0; i < 6; i++) {inF.elems[i] = i; w.elems[i] = i; wGrads.elems[i] = i; outB.elems[i] = i;}
    for (int i = 0; i < 4; i++) {outF.elems[i] = i; inB.elems[i] = i;}
    for (int i = 0; i < 2; i++) b.elems[i] = i;
    
    float outFDesiredElems[] = {
        5, 15,
        14, 51
    };

    float outBDesiredElems[] = {
        6, 8, 10,
        9, 13, 17
    };

    float wGradsDesiredElems[] = {
        3, 4, 5,
        9, 14, 19
    };

    outFDesired.elems = outFDesiredElems;
    outBDesired.elems = outBDesiredElems;
    wGradsDesired.elems = wGradsDesiredElems;

    OperationDenseForward(inF, w, b, outF);
    if (!VecArrIsEqual(outF, outFDesired)) TEST_FAILED
    OperationDenseBackward(inF, inB, w, wGrads, outB);
    if (!VecArrIsEqual(wGrads, wGradsDesired)) TEST_FAILED
    if (!VecArrIsEqual(outB, outBDesired)) TEST_FAILED
    TEST_PASSED
}

TEST(TestOperationReluForwardAndBackward)() {
    struct vecArr inF = VecArrCreate(3, 2);
    struct vecArr inB = VecArrCreate(3, 2);
    struct vecArr outF = VecArrCreate(3, 2);
    struct vecArr outFDesired = VecArrCreate(3, 2);
    struct vecArr outB = VecArrCreate(3, 2);
    struct vecArr outBDesired = VecArrCreate(3, 2);
    for (int i = 0; i < 6; i++) {
        inF.elems[i] = i -2;
        inB.elems[i] = i;
        outF.elems[i] = i -2;
        outB.elems[i] = i -2;
    }

    float outFDesiredElems[] = {0, 0, 0, 1, 2, 3};
    float outBDesiredElems[] = {0, 0, 0, 3, 4, 5};
    outFDesired.elems = outFDesiredElems;
    outBDesired.elems = outBDesiredElems;
    OperationReluForward(inF, outF);
    if (!VecArrIsEqual(outF, outFDesired)) TEST_FAILED
    OperationReluBackward(inF, inB, outB);
    if (!VecArrIsEqual(outB, outBDesired)) TEST_FAILED
    TEST_PASSED
}
