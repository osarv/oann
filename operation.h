#ifndef OPERATION_H
#define OPERATION_H
#include "vecarr.h"

void OperationReluForward(struct vecArr x, struct vecArr y);
void OperationReluBackward(struct vecArr x, struct vecArr dy, struct vecArr dx);

//operations header for collected operations wrapping computation methods
void OperationDenseForward(struct vecArr x, struct vecArr w, struct vecArr b, struct vecArr y);
void OperationDenseBackward(struct vecArr x, struct vecArr dy, struct vecArr w, struct vecArr wGrads, struct vecArr dx);
void OperationSCEForward(struct vecArr x, struct vecArr y);
void OperationSCEBackward(struct vecArr y, struct vecArr labels, struct vecArr dx);
float OperationSCECalcLoss(struct vecArr y, struct vecArr labels);

#endif //OPERATION_H
