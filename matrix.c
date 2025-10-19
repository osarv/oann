#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"
#include "util.h"

struct matrix {
    int nRows;
    int nCols;
    int rowStride;
    int colStride;
    float* elems;
};

Matrix MatrixCreate(int nRows, int nCols, enum matrixOrder order) {
    Matrix m = malloc(sizeof(struct matrix));
    CheckPtr(m);
    m->elems = malloc(sizeof(float) * nRows * nCols);
    CheckPtr(m->elems);
    m->nRows = nRows;
    m->nCols = nCols;
    m->rowStride = order == ROW_MAJOR ? 1 : nCols;
    m->colStride = order == COL_MAJOR ? 1 : nRows;
    return m;
}

void MatrixInitConst(Matrix m, float c) {
    for (int i = 0; i < m->nRows * m->nCols; i++) {
        m->elems[i] = c;
    }
}
