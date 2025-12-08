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
    if (a->m.elems) VecArrDestroy(a->m); //check needed for recipe optimizer
    if (a->v.elems) VecArrDestroy(a->v); //check needed for recipe optimizer
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
    struct adamw* a = CallocOrCrash(sizeof(struct adamw)); //calloc needed to set m and v elems ptrs to NULL
    a->o.optimize = AdamWOptimize;
    a->o.destroy = AdamWDestroy;
    a->o.createCopy = AdamWCreateCopy;
    a->lr = lr;
    a->mDecay = mDecay;
    a->vDecay = vDecay;
    a->wDecay = wDecay;
    return &a->o;
}
