#ifndef VECARR_H
#define VECARR_H
#include "util.h"

struct vecArr {
    int vecLen;
    int nVecs;
    float* elems;
};

struct vecArr VecArrCreate(int vecLen, int nVec);
void VecArrDestroy(struct vecArr v);
struct list VecArrListInit();
void VecArrListAdd(struct list* l, struct vecArr va);
bool VecArrIsEqual(struct vecArr va1, struct vecArr va2);
void VecArrPrint(struct vecArr va);

#endif //VECARR_H
