#ifndef VECARR_H
#define VECARR_H
#include "util.h"

struct vecArr {
    int vecLen;
    int nVecs;
    float* elems;
};

struct vecArr VecArrCreate(int vecLen, int nVec);
struct vecArr VecArrCreateSameDim(struct vecArr v);
void VecArrDestroy(struct vecArr v);
void VecArrInitConst(struct vecArr v, float c);
void VecArrInitUniform(struct vecArr v, float min, float max);
void VecArrInitNormDist(struct vecArr v, float mean, float stddev);
struct list VecArrListInit();
void VecArrListAdd(struct list* l, struct vecArr va);
bool VecArrIsEqual(struct vecArr va1, struct vecArr va2);
void VecArrPrint(struct vecArr va);

#endif //VECARR_H
