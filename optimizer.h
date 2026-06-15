#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "vecarr.h"
#include "oann.h"

//everything is owned and freed by the network
struct optimizer {
    void (*optimize)(Optimizer o, VecArr p, VecArr dp);
    Optimizer (*yieldCopy)(Optimizer oRecipe, VecArr p);
};

#endif //OPTIMIZER_H
