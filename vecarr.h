#ifndef VECARR_H
#define VECARR_H
#include "util.h"

typedef float* VecArr;
int VecArrVecLen(VecArr v);
int VecArrNVecs(VecArr v);
int VecArrNElems(VecArr v);
VecArr VecArrCreate(int vecLen, int nVec);
VecArr VecArrCreateSameDim(VecArr v);
void VecArrDestroy(VecArr v);
void VecArrInitConst(VecArr v, float c);
void VecArrInitUniform(VecArr v, float min, float max);
void VecArrInitNormDist(VecArr v, float mean, float stddev);
bool VecArrFloatArrIsEqual(VecArr v, float* fArr);
void VecArrPrint(VecArr v);

#endif //VECARR_H
