#ifndef VECARR_H
#define VECARR_H
#include "oann.h"
#include "util.h"

typedef OANNfloat* VecArr;
int VecArrVecLen(VecArr v);
int VecArrNVecs(VecArr v);
int VecArrNElems(VecArr v);
VecArr VecArrCreate(int vecLen, int nVec);
VecArr VecArrCreateSameDim(VecArr v);
void VecArrDestroy(VecArr v);
void VecArrInitConst(VecArr v, OANNfloat c);
void VecArrInitUniform(VecArr v, OANNfloat min, OANNfloat max);
void VecArrInitNormDist(VecArr v, OANNfloat mean, OANNfloat stddev);
OANNfloat VecArrMean(VecArr v);
OANNfloat VecArrStdDev(VecArr v);
void VecArrNormalize(VecArr v);
bool VecArrIsNan(VecArr v);
bool VecArrOANNfloatArrIsEqual(VecArr v, OANNfloat* fArr);
void VecArrPrint(VecArr v);

#endif //VECARR_H
