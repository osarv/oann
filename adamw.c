#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "optimizer.h"
#include "util.h"
//adaptive momentum with l2 weight decay

struct adamw {
    struct optimizer o;
    OANNfloat lr;
    OANNfloat mDecay;
    OANNfloat vDecay;
    OANNfloat wDecay;
    VecArr m;
    VecArr v;
};

#define ADAMW_NUM_STAB_CONST 1e-8
static void adamWOptimize(Optimizer o, VecArr p, VecArr dp) {
    struct adamw* a = (struct adamw*)o;

    for (int i = 0; i < VecArrNElems(p); i++) {
        a->m[i] = a->mDecay * a->m[i] + (1 - a->mDecay) * dp[i];
        a->v[i] = a->vDecay * a->v[i] + (1 - a->vDecay) * dp[i] * dp[i];
        p[i] -= a->lr * (a->m[i] / (a->v[i] + ADAMW_NUM_STAB_CONST) + a->wDecay * p[i]);
    }
}

static void adamWDestroy(Optimizer o) {
    struct adamw* a = (struct adamw*)o;
    if (a->m) VecArrDestroy(a->m); //check needed for recipe optimizer
    if (a->v) VecArrDestroy(a->v); //check needed for recipe optimizer
    free(a);
}

Optimizer AdamWCreate(OANNfloat lr, OANNfloat mDecay, OANNfloat vDecay, OANNfloat wDecay);
static Optimizer adamWYieldOptimizer(Optimizer o, VecArr p) {
    struct adamw* recipe = (struct adamw*)o;
    struct adamw* new = (struct adamw*)AdamWCreate(recipe->lr, recipe->mDecay, recipe->vDecay, recipe->wDecay);
    new->m = VecArrCreateSameDim(p);
    new->v = VecArrCreateSameDim(p);
    VecArrInitConst(new->m, 0);
    VecArrInitConst(new->v, 0);
    return &new->o;
}

Optimizer AdamWCreate(OANNfloat lr, OANNfloat mDecay, OANNfloat vDecay, OANNfloat wDecay) {
    struct adamw* a = CallocOrCrash(sizeof(struct adamw)); //calloc needed to set m and v to NULL
    a->o.optimize = adamWOptimize;
    a->o.destroy = adamWDestroy;
    a->o.yieldOptimizer = adamWYieldOptimizer;
    a->lr = lr;
    a->mDecay = mDecay;
    a->vDecay = vDecay;
    a->wDecay = wDecay;
    return &a->o;
}

TEST(AdamW) {
    OANNfloat lr = 0.01;
    OANNfloat mDecay = 0.9;
    OANNfloat vDecay = 0.99;
    OANNfloat wDecay = 0.01;
    Optimizer oRecipe = AdamWCreate(lr, mDecay, vDecay, wDecay);
    VecArr p = VecArrCreate(2, 3);
    VecArr dp = VecArrCreateSameDim(p);
    Optimizer o = oRecipe->yieldOptimizer(oRecipe, p);
    struct adamw* a = (struct adamw*)o;
    o->destroy(oRecipe);
    OANNfloat mDesired[6];
    OANNfloat vDesired[6];
    OANNfloat pDesired[6];
    for (int i = 0; i < 2 * 3; i++) {
        p[i] = i;
        dp[i] = i;
        a->m[i] = i;
        a->v[i] = i;
        mDesired[i] = mDecay * i + (1 - mDecay) * i;
        vDesired[i] = vDecay * i + (1 - vDecay) * i * i;
        pDesired[i] = i - lr * (mDesired[i] / (vDesired[i] + ADAMW_NUM_STAB_CONST) + wDecay * i);
    }
    o->optimize(o, p, dp);
    if (!VecArrOANNfloatArrIsEqual(a->m, mDesired)) TEST_FAILED
    if (!VecArrOANNfloatArrIsEqual(a->v, vDesired)) TEST_FAILED
    if (!VecArrOANNfloatArrIsEqual(p, pDesired)) TEST_FAILED
    o->destroy(o);
    TEST_PASSED
}
