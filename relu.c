#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
#include "operation.h"

void ReluMount(Layer l, struct vecArr x) {
    l->y = VecArrCreateSameDim(x);
    l->dy = VecArrCreateSameDim(x);
}

void ReluInit(Layer l, struct vecArr x) {
    (void)l; //correct implementation
    (void)x; //correct implementation
}

void ReluDestroy(Layer l) {
    free(l);
}

void ReluForward(Layer l, struct vecArr x) {
    OperationReluForward(x, l->y);
}

void ReluBackward(Layer l, struct vecArr x, struct vecArr dx) {
    OperationReluBackward(x, l->dy, dx);
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
