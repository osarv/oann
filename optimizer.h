#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "vecarr.h"
#include "oann.h"

struct optimizer {
    void (*optimize)(Optimizer o, struct vecArr optParams, struct vecArr dOptParams);
    Optimizer (*initCopy)(Optimizer o, struct vecArr optParams);
};

void OptimizerOptimize(Optimizer o, struct vecArr optParams, struct vecArr dOptParams);
Optimizer OptimizerInitCopy(Optimizer o, struct vecArr optParams);

#endif //OPTIMIZER_H
