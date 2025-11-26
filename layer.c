#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include "layer.h"
#include "util.h"
#include "vecarr.h"

void LayerDestroy(Layer l) {
    l->destroy(l);
}

void LayerInit(Layer l, struct vecArr inF) {
    l->init(l, inF);
}

void LayerMount(Layer l, struct vecArr inF) {
    l->optParams = VecArrListInit();
    l->optParamGrads = VecArrListInit();
    l->mount(l, inF);
}

void LayerForward(Layer l, struct vecArr inF) {
    l->forward(l, inF);
}

void LayerBackward(Layer l, struct vecArr inF, struct vecArr inB) {
    l->backward(l, inF, inB);
}
