#include <stdlib.h>
#include <stdio.h>
#include "optimizer.h"

void OptimizerOptimize(Optimizer o, struct vecArr p, struct vecArr dp) {
    o->optimize(o, p, dp);
}

void OptimizerDestroy(Optimizer o) {
    o->destroy(o);
}

Optimizer OptimizerCreateCopy(Optimizer o, struct vecArr p) {
    return o->createCopy(o, p);
}
