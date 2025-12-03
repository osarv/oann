#include <stdlib.h>
#include <stdio.h>
#include "losser.h"
#include "operation.h"
//softmax cross entropy loss

void SCEForward(Losser l, struct vecArr x) {
    OperationSCEForward(x, l->y);
}

void SCEBackward(Losser l, struct vecArr x, struct vecArr labels, struct vecArr dx) {
    (void)x; //correct implementation
    OperationSCEBackward(l->y, labels, dx);
}

float SCECalcLoss(Losser l, struct vecArr x, struct vecArr labels) {
    (void)x; //correct implementation
    return OperationSCECalcLoss(l->y, labels);
}


Losser SCECreate() {
    struct losser* s = MallocOrCrash(sizeof(struct losser));
    s->forward = SCEForward;
    s->backward = SCEBackward;
    s->calcLoss = SCECalcLoss;
    return s;
}
