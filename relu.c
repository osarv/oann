#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
#include "operation.h"

void ReluUnmount(Layer l) {
    VecArrDestroy(l->y);
    VecArrDestroy(l->dy);
}

void ReluMount(Layer l, struct vecArr x) {
    l->y = VecArrCreateSameDim(x);
    l->dy = VecArrCreateSameDim(x);
}

void ReluInit(Layer l, int nIn) {
    (void)l; //correct implementation
    (void)nIn; //correct implementation
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
    Layer l = CallocOrCrash(sizeof(struct layer)); //calloc needed to initialized lists to zero length
    l->unmount = ReluUnmount;
    l->mount = ReluMount;
    l->init = ReluInit;
    l->destroy = ReluDestroy;
    l->forward = ReluForward;
    l->backward = ReluBackward;
    return l;
}
