#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <cblas.h>
#include "operation.h"
#include "util.h"

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

#ifdef TEST
void TestOperationDenseForwardAndBackward() {
    struct vecArr inF = VecArrCreate(3, 2);
    struct vecArr w = VecArrCreate(3, 2);
    struct vecArr b = VecArrCreate(2, 1);
    struct vecArr outF = VecArrCreate(2, 2);
    struct vecArr outFDesired = VecArrCreate(2, 2);
    struct vecArr wGrads = VecArrCreate(3, 2);
    struct vecArr wGradsDesired = VecArrCreate(3, 2);
    struct vecArr inB = VecArrCreate(2, 2);
    struct vecArr outB = VecArrCreate(3, 2);
    struct vecArr outBDesired = VecArrCreate(3, 2);
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
#endif //TEST

#endif //OP_MODE_BLAS
