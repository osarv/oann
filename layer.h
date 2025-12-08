#ifndef LAYER_H
#define LAYER_H
#include "oann.h"
#include "vecarr.h"
#include "util.h"

//All layer implementations must include the layer struct, preferably as the first member.
//Additionally, they must implement the mount, init, destroy, forward backward functions.
//Furthermore they must and add any parameters and gradients to be optimized to the p and dp lists

struct layer {
    struct vecArr y;
    struct vecArr dy;
    void (*mount)(Layer l, struct vecArr x);
    void (*init)(Layer l, struct vecArr x);
    void (*destroy)(Layer l);
    void (*forward)(Layer l, struct vecArr x);
    void (*backward)(Layer l, struct vecArr x, struct vecArr dx);
    struct list p; //list of parameters to optimize
    struct list dp; //list of gradients of parameters to optimize
    struct list optimizers; //corresponding optimizers
};

#endif //LAYER_H
