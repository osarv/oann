#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "operation.h"

struct dense {
    struct layer l;
    struct vecArr w;
    struct vecArr dw;
    struct vecArr b;
    struct vecArr db;
};

void DenseUnmount(Layer l) {
    VecArrDestroy(l->y);
    VecArrDestroy(l->dy);
}

void DenseMount(Layer l, struct vecArr x) {
    l->y = VecArrCreate(l->nOut, x.nVecs);
    l->dy = VecArrCreate(l->nOut, x.nVecs);
}

void DenseInit(Layer l, int nIn) {
    struct dense* d = (struct dense*)l;
    d->w = VecArrCreate(nIn, l->nOut);
    d->dw = VecArrCreateSameDim(d->w);
    d->b = VecArrCreate(l->nOut, 1);
    d->db = VecArrCreateSameDim(d->db);
    l->p = VecArrListInit();
    l->dp = VecArrListInit();
    VecArrListAdd(&l->p, d->w);
    VecArrListAdd(&l->p, d->b);
    VecArrListAdd(&l->dp, d->dw);
    VecArrListAdd(&l->dp, d->db);
}

void DenseDestroy(Layer l) {
    struct dense* d = (struct dense*)l;
    VecArrDestroy(d->w);
    VecArrDestroy(d->b);
    ListDestroy(l->p);
    ListDestroy(l->dp);
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
    struct dense* d = CallocOrCrash(sizeof(struct dense)); //calloc needed to initialized lists to zero length
    d->l.nOut = nOut;
    d->l.unmount = DenseUnmount;
    d->l.mount = DenseMount;
    d->l.init = DenseInit;
    d->l.destroy = DenseDestroy;
    d->l.forward = DenseForward;
    d->l.backward = DenseBackward;
    return &d->l;
}
