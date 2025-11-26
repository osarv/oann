#ifndef OPERATION_H
#define OPERATION_H
#include "vecarr.h"

void OperationReluForward(struct vecArr inF, struct vecArr outF);
void OperationReluBackward(struct vecArr inF, struct vecArr inB, struct vecArr outB);

//operations header for collected operations wrapping computation methods
void OperationDenseForward(struct vecArr inF, struct vecArr w, struct vecArr b, struct vecArr outF);
void OperationDenseBackward(struct vecArr inF, struct vecArr inB, struct vecArr w, struct vecArr wGrads, struct vecArr outB);

#endif //OPERATION_H
