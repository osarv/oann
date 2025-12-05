#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "operation.h"

struct dense {
    struct layer l;
    int nOut;
    struct vecArr w;
    struct vecArr dw;
    struct vecArr b;
    struct vecArr db;
};

void DenseMount(Layer l, struct vecArr x) {
    struct dense* d = (struct dense*)l;
    l->y = VecArrCreate(d->nOut, x.nVecs);
    l->dy = VecArrCreate(d->nOut, x.nVecs);
    VecArrListAdd(&l->optParams, d->w);
    VecArrListAdd(&l->optParams, d->b);
    VecArrListAdd(&l->dOptParams, d->dw);
    VecArrListAdd(&l->dOptParams, d->db);
}

void DenseInit(Layer l, struct vecArr x) {
    struct dense* d = (struct dense*)l;
    d->w = VecArrCreate(x.vecLen, d->nOut);
    d->b = VecArrCreate(d->nOut, 1);
}

void DenseDestroy(Layer l) {
    struct dense* d = (struct dense*)l;
    VecArrDestroy(d->w);
    VecArrDestroy(d->b);
    free(d);
}

void DenseForward(Layer l, struct vecArr x) {
    struct dense* d = (struct dense*)l;
    OperationDenseForward(x, d->w, d->b, l->y);
}

void DenseBackward(Layer l, struct vecArr x, struct vecArr dx) {
    struct dense* d = (struct dense*)l;
    OperationDenseBackward(x, l->dy, d->w, d->dw, dx);
}

Layer DenseCreate(int nOut) {
    struct dense* d = MallocOrCrash(sizeof(struct dense));
    d->nOut = nOut;
    d->l.mount = DenseMount;
    d->l.init = DenseInit;
    d->l.destroy = DenseDestroy;
    d->l.forward = DenseForward;
    d->l.backward = DenseBackward;
    return &d->l;
}
