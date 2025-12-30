#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vecarr.h"
#include "util.h"

int VecArrVecLen(VecArr v) {
    return ((int*)v)[-2];
}

int VecArrNVecs(VecArr v) {
    return ((int*)v)[-1];
}

int VecArrNElems(VecArr v) {
    return VecArrVecLen(v) * VecArrNVecs(v);
}

VecArr VecArrCreate(int vecLen, int nVecs) {
    VecArr v = MallocOrCrash(2 * sizeof(int) + sizeof(float) * vecLen * nVecs);
    ((int*)v)[0] = vecLen;
    ((int*)v)[1] = nVecs;
    return (float*)(((int*)v) + 2);
}

VecArr VecArrCreateSameDim(VecArr v) {
    return VecArrCreate(VecArrVecLen(v), VecArrNVecs(v));
}

void VecArrDestroy(VecArr v) {
    free(((int*)v) - 2);
}

void VecArrInitConst(VecArr v, float c) {
    for (int i = 0; i < VecArrNElems(v); i++) {
        v[i] = c;
    }
}

static float getRandomFloat(float min, float max) {
    return ((float)(rand()%1000000000)) / 1000000000.0f * (max - min) - min;
}

void VecArrInitUniform(VecArr v, float min, float max) {
    for (int i = 0; i < VecArrNElems(v); i++) {
        v[i] = getRandomFloat(min, max);
    }
}

static void getStdDevPair(float* a, float* b) {
    float x = getRandomFloat(-1,1);
    float y = getRandomFloat(-1,1);
    float s = x * x + y * y;
    float lnS = log(s);
    *a = x * sqrt(-2 * lnS / s);
    *b = y * sqrt(-2 * lnS / s);
}

void VecArrInitNormDist(VecArr v, float mean, float stddev) {
    for (int i = 0; i < (VecArrNElems(v) - VecArrNElems(v)%2) / 2; i++) {
        getStdDevPair(&v[i], &v[i +1]);
        v[i] = (v[i] + mean) * stddev;
        v[i +1] = (v[i +1] + mean) * stddev;
    }
    getStdDevPair(&v[VecArrNElems(v) -1], &v[VecArrNElems(v) -1]);
    v[VecArrNElems(v) -1] = (v[VecArrNElems(v) -1] + mean) * stddev;
}

bool VecArrFloatArrIsEqual(VecArr v, float* fArr) {
    for (int i = 0; i < VecArrNElems(v); i++) {
        if (v[i] != fArr[i]) return false;
    }
    return true;
}

void VecArrPrint(VecArr v) {
    printf("vector array: vecLen=%d, nVecs =%d\n", VecArrVecLen(v), VecArrNVecs(v));
    for (int i = 0; i < VecArrNVecs(v); i++) {
        for (int j = 0; j < VecArrVecLen(v); j++) {
            printf("%f, ", v[i * VecArrVecLen(v) + j]);
        }
        puts("");
    }
}
