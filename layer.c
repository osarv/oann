#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include "layer.h"
#include "util.h"
#include "vecarr.h"

void LayerDestroy(Layer l) {
    l->destroy(l);
}

void LayerInit(Layer l, struct vecArr x) {
    l->init(l, x);
}

void LayerMount(Layer l, struct vecArr x) {
    l->optParams = VecArrListInit();
    l->dOptParams = VecArrListInit();
    l->mount(l, x);
}

void LayerForward(Layer l, struct vecArr x) {
    l->forward(l, x);
}

void LayerBackward(Layer l, struct vecArr x, struct vecArr dy) {
    l->backward(l, x, dy);
}
