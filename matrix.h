#ifndef MATRIX_H
#define MATRIX_H

typedef struct matrix* Matrix;

enum matrixOrder {
    ROW_MAJOR,
    COL_MAJOR
};

Matrix MatrixCreate(int nRows, int nCols, enum matrixOrder order);
void MatrixInitConst(Matrix m, float c);

#endif //MATRIX_H
