#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "vecarr.h"
#include "oann.h"

struct optimizer {
    void (*optimize)(Optimizer o, VecArr p, VecArr dp);
    void (*destroy)(Optimizer o); //call with recipe optimizer is defined
    Optimizer (*yieldOptimizer)(Optimizer oRecipe, VecArr p);
};

#endif //OPTIMIZER_H
