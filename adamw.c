#include <stdlib.h>
#include <stdio.h>
#include "optimizer.h"
#include "util.h"
#include "operation.h"

struct adamw {
    struct optimizer o;
    float lr;
    float mDecay;
    float vDecay;
    float wDecay;
    struct vecArr m;
    struct vecArr v;
};

void AdamWOptimize(Optimizer o, struct vecArr p, struct vecArr dp) {
    struct adamw* a = (struct adamw*)o;
    OperationAdamWOptimize(p, dp, a->m, a->v, a->lr, a->mDecay, a->vDecay, a->wDecay);
}

void AdamWDestroy(Optimizer o) {
    struct adamw* a = (struct adamw*)o;
    VecArrDestroy(a->m);
    VecArrDestroy(a->v);
    free(a);
}

Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay);
Optimizer AdamWCreateCopy(Optimizer o, struct vecArr p) {
    struct adamw* recipe = (struct adamw*)o;
    struct adamw* new = (struct adamw*)AdamWCreate(recipe->lr, recipe->mDecay, recipe->vDecay, recipe->wDecay);
    new->m = VecArrCreateSameDim(p);
    new->v = VecArrCreateSameDim(p);
    VecArrInitConst(new->m, 0);
    VecArrInitConst(new->v, 0);
    return &new->o;
}

Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay) {
    struct adamw* a = MallocOrCrash(sizeof(struct adamw));
    a->o.optimize = AdamWOptimize;
    a->o.destroy = AdamWDestroy;
    a->o.createCopy = AdamWCreateCopy;
    a->lr = lr;
    a->mDecay = mDecay;
    a->vDecay = vDecay;
    a->wDecay = wDecay;
    return &a->o;
}
