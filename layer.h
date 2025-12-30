#ifndef LAYER_H
#define LAYER_H
#include "oann.h"
#include "vecarr.h"
#include "util.h"

//All layer implementations must include the layer struct, preferably as the first member.
//Additionally, they must implement the mount, init, destroy, forward backward functions.
//Furthermore they must and add any parameters and gradients to be optimized to the p and dp lists

struct layer {
    int nOut;
    VecArr y;
    VecArr dy;
    void (*unmount)(Layer l);
    void (*mount)(Layer l, VecArr x);
    void (*init)(Layer l, int nIn);
    void (*destroy)(Layer l); //presumes former unmounting
    void (*forward)(Layer l, VecArr x);
    void (*backward)(Layer l, VecArr x, VecArr dx); //does NOT call the optimizers
    struct ptrList p; //list of parameters to optimize
    struct ptrList dp; //list of gradients of parameters to optimize
    struct ptrList optimizers; //corresponding optimizers
};

#endif //LAYER_H
