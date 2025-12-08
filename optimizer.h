#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "vecarr.h"
#include "oann.h"

struct optimizer {
    void (*optimize)(Optimizer o, struct vecArr p, struct vecArr dp);
    void (*destroy)(Optimizer o); //do not call for recipe optimizer
    Optimizer (*createCopy)(Optimizer o, struct vecArr p);
};

#endif //OPTIMIZER_H
