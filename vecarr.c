#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "vecarr.h"
#include "util.h"

struct vecArr VecArrCreate(int vecLen, int nVecs) {
    struct vecArr v;
    v.elems = MallocOrCrash(sizeof(float) * vecLen * nVecs);
    v.vecLen = vecLen;
    v.nVecs = nVecs;
    return v;
}

void VecArrDestroy(struct vecArr v) {
    free(v.elems);
}

void VecArrInitConst(struct vecArr v, float c) {
    for (int i = 0; i < v.vecLen * v.nVecs; i++) {
        v.elems[i] = c;
    }
}

static float getRandomFloat(float min, float max) {
    return ((float)(rand()%1000000000)) / 1000000000.0f * (max - min) - min;
}

void VecArrInitUniform(struct vecArr v, float min, float max) {
    for (int i = 0; i < v.vecLen * v.nVecs; i++) {
        v.elems[i] = getRandomFloat(min, max);
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

void VecArrInitNormDist(struct vecArr v, float mean, float stddev) {
    int elemLen = v.vecLen * v.nVecs;
    for (int i = 0; i < (elemLen - elemLen%2) / 2; i++) {
        getStdDevPair(&v.elems[i], &v.elems[i +1]);
        v.elems[i] = (v.elems[i] + mean) * stddev;
        v.elems[i +1] = (v.elems[i +1] + mean) * stddev;
    }
    getStdDevPair(&v.elems[elemLen -1], &v.elems[elemLen -1]);
    v.elems[elemLen -1] = (v.elems[elemLen -1] + mean) * stddev;
}

struct list VecArrListInit() {
    return ListInit(sizeof(struct vecArr));
}

void VecArrListAdd(struct list* l, struct vecArr va) {
    ListAdd(l, &va);
}

bool VecArrIsEqual(struct vecArr va1, struct vecArr va2) {
    if (va1.vecLen != va2.vecLen) return false;
    if (va1.nVecs != va2.nVecs) return false;
    for (int i = 0; i < va1.vecLen * va1.nVecs; i++) {
        if (va1.elems[i] != va2.elems[i]) return false;
    }
    return true;
}

void VecArrPrint(struct vecArr va) {
    printf("vector array: vecLen=%d, nVecs =%d\n", va.vecLen, va.nVecs);
    for (int i = 0; i < va.nVecs; i++) {
        for (int j = 0; j < va.vecLen; j++) {
            printf("%f, ", va.elems[i * va.vecLen + j]);
        }
        puts("");
    }
}
