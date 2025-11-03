#ifndef MATRIX_H
#define MATRIX_H

#include <stdint.h>

//no boundary checking or matrix compatibility is done

typedef struct matrix* Matrix;

struct matrix {
    int nRows;
    int nCols;
    float* elems;
};

Matrix MatrixCreate(int nRows, int nCols);
void MatrixInitConst(Matrix m, float c);
void MatrixInitUniform(Matrix m, float min, float max);
void MatrixInitNormDist(Matrix m, float mean, float stddev);

#endif //MATRIX_H
