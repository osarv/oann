#include <stdlib.h>
#include <stdio.h>
#include "layer.h"
#include "vecarr.h"
#include "util.h"
#include "operation.h"

struct dense {
    struct layer l;
    int nOut;
    struct vecArr w;
    struct vecArr wGrads;
    struct vecArr b;
    struct vecArr bGrads;
};

void DenseMount(Layer l, struct vecArr inF) {
    struct dense* d = (struct dense*)l;
    l->outF = VecArrCreate(d->nOut, inF.nVecs);
    l->inB = VecArrCreate(d->nOut, inF.nVecs);
    l->optParams = VecArrListInit();
    VecArrListAdd(&l->optParams, d->w);
    VecArrListAdd(&l->optParams, d->b);
}

void DenseInit(Layer l, struct vecArr inF) {
    struct dense* d = (struct dense*)l;
    d->w = VecArrCreate(inF.vecLen, d->nOut);
    d->b = VecArrCreate(d->nOut, 1);
}

void DenseDestroy(Layer l) {
    struct dense* d = (struct dense*)l;
    VecArrDestroy(d->w);
    VecArrDestroy(d->b);
    free(d);
}

void DenseForward(Layer l, struct vecArr inF) {
    struct dense* d = (struct dense*)l;
    OperationDenseForward(inF, d->w, d->b, l->outF);
}

void DenseBackward(Layer l, struct vecArr inF, struct vecArr outB) {
    struct dense* d = (struct dense*)l;
    OperationDenseBackward(inF, l->inB, d->w, d->wGrads, outB);
}

Layer DenseCreate(int nOut) {
    struct dense* d = MallocOrCrash(sizeof(struct dense));
    d->nOut = nOut;
    d->l.mount = DenseMount;
    d->l.init = DenseInit;
    d->l.destroy = DenseDestroy;
    d->l.forward = DenseForward;
    d->l.backward = DenseBackward;
    return &(d->l);
}
