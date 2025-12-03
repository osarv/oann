#include <stdlib.h>
#include <stdio.h>
#include "optimizer.h"

void OptimizerOptimize(Optimizer o, struct vecArr optParams, struct vecArr dOptParams) {
    o->optimize(o, optParams, dOptParams);
}
