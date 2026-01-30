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
    return ((float)(rand()%1000000001)) / 1000000000.0f * (max - min) + min;
}

void VecArrInitUniform(VecArr v, float min, float max) {
    for (int i = 0; i < VecArrNElems(v); i++) {
        v[i] = getRandomFloat(min, max);
    }
}

static void getStdDevPair(float* a, float* b) {
    float x, y, s;
    do {
        x = getRandomFloat(-1,1);
        y = getRandomFloat(-1,1);
        s = x * x + y * y;
    }
    while (s == 0 || s >= 1);
    float lnS = log(s);
    *a = x * sqrt(-2 * lnS / s);
    *b = y * sqrt(-2 * lnS / s);
}

void VecArrInitNormDist(VecArr v, float mean, float stddev) {
    for (int i = 0; i < VecArrNElems(v) - 1; i += 2) {
        getStdDevPair(&v[i], &v[i +1]);
        v[i] = v[i] * stddev + mean;
        v[i +1] = v[i +1] * stddev + mean;
    }
    if (VecArrNElems(v) % 2) {
        getStdDevPair(&v[VecArrNElems(v) -1], &v[VecArrNElems(v) -1]);
        v[VecArrNElems(v) -1] = v[VecArrNElems(v) -1] * stddev + mean;
    }
}

float VecArrMean(VecArr v) {
    float sum = 0;
    for (int i = 0; i < VecArrNElems(v); i++) {
        sum += v[i];
    }
    return sum / VecArrNElems(v);
}

float VecArrStdDev(VecArr v) {
    float mean = VecArrMean(v);
    float sum = 0;
    for (int i = 0; i < VecArrNElems(v); i++) {
        sum += (v[i] - mean) * (v[i] - mean);
    }
    return sqrt(sum / VecArrNElems(v));
}

bool VecArrIsNan(VecArr v) {
    for (int i = 0; i < VecArrNElems(v); i++) {
        if (isnan(v[i])) return true;
    }
    return false;
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
