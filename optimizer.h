#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "vecarr.h"
#include "oann.h"

struct optimizer {
    void (*optimize)(Optimizer o, struct vecArr p, struct vecArr dp);
    void (*destroy)(Optimizer o); //call with recipe optimizer is defined
    Optimizer (*createCopy)(Optimizer o, struct vecArr p);
};

#endif //OPTIMIZER_H
