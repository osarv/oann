#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
#include "operation.h"

void ReluMount(Layer l, struct vecArr inF) {
    l->outF = VecArrCreate(inF.vecLen, inF.nVecs);
    l->inB = VecArrCreate(inF.vecLen, inF.nVecs);
}

void ReluInit(Layer l, struct vecArr inF) {
    (void)l; //correct implementation
    (void)inF; //correct implementation
}

void ReluDestroy(Layer l) {
    free(l);
}

void ReluForward(Layer l, struct vecArr inF) {
    OperationReluForward(inF, l->inB);
}

void ReluBackward(Layer l, struct vecArr inF, struct vecArr outB) {
    OperationReluBackward(inF, l->inB, outB);
}

Layer ReluCreate() {
    Layer l = MallocOrCrash(sizeof(struct layer));
    l->mount = ReluMount;
    l->init = ReluInit;
    l->destroy = ReluDestroy;
    l->forward = ReluForward;
    l->backward = ReluBackward;
    return l;
}
