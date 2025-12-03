#include <stdlib.h>
#include <stdio.h>
#include "losser.h"

void LosserForward(Losser l, struct vecArr x) {
    l->forward(l, x);
}

void LosserBackward(Losser l, struct vecArr x, struct vecArr labels, struct vecArr dx) {
    l->backward(l, x, labels, dx);
}

float LosserCalcLoss(Losser l, struct vecArr x, struct vecArr labels) {
    return l->calcLoss(l, x, labels);
}

