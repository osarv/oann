#ifndef OPERATION_H
#define OPERATION_H
#include "vecarr.h"

void OperationReluForward(VecArr x, VecArr y);
void OperationReluBackward(VecArr x, VecArr dy, VecArr dx);

//operations header for collected operations wrapping computation methods
void OperationDenseForward(VecArr x, VecArr w, VecArr b, VecArr y);
void OperationDenseBackward(VecArr x, VecArr dy, VecArr w, VecArr wGrads, VecArr dx);
void OperationSCEForward(VecArr x, VecArr y);
void OperationSCEBackward(VecArr y, VecArr labels, VecArr dx);
float OperationSCECalcLoss(VecArr y, VecArr labels);
void OperationAdamWOptimize(VecArr p, VecArr dp, VecArr m, VecArr v,
        float lr, float mDecay, float vDecay, float wDecay);

#endif //OPERATION_H
