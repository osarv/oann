#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>
#include "matrix.h"
#include "util.h"

Matrix MatrixCreate(int nRows, int nCols) {
    Matrix m = MallocOrCrash(sizeof(struct matrix));
    m->elems = MallocOrCrash(sizeof(float) * nRows * nCols);
    m->nRows = nRows;
    m->nCols = nCols;
    return m;
}

void MatrixInitConst(Matrix m, float c) {
    for (int i = 0; i < m->nRows * m->nCols; i++) {
        m->elems[i] = c;
    }
}

static float getRandomFloat(float min, float max) {
    return ((float)(rand()%1000000000)) / 1000000000.0f * (max - min) - min;
}

void MatrixInitUniform(Matrix m, float min, float max) {
    for (int i = 0; i < m->nRows * m->nCols; i++) {
        m->elems[i] = getRandomFloat(min, max);
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

void MatrixInitNormDist(Matrix m, float mean, float stddev) {
    int elemLen = m->nRows * m->nCols;
    for (int i = 0; i < (elemLen - elemLen%2) / 2; i++) {
        getStdDevPair(&m->elems[i], &m->elems[i +1]);
        m->elems[i] = (m->elems[i] + mean) * stddev;
        m->elems[i +1] = (m->elems[i +1] + mean) * stddev;
    }
    getStdDevPair(&m->elems[elemLen -1], &m->elems[elemLen -1]);
    m->elems[elemLen -1] = (m->elems[elemLen -1] + mean) * stddev;
}

void MatrixMul(Matrix a, Matrix b, Matrix res, bool aTrans, bool bTrans) {
    cblas_sgemm(CblasRowMajor, aTrans ? CblasTrans : CblasNoTrans, bTrans ? CblasTrans : CblasNoTrans,
            a->nRows, b->nCols, b->nCols, 1.0, a->elems, 1, b->elems, 1, 1.0, res->elems, 1);
}
