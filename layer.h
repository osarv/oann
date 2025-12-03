#ifndef LAYER_H
#define LAYER_H
#include "oann.h"
#include "vecarr.h"
#include "util.h"

//All layer implementations must include the layer struct, preferably as the first member.
//Additionally, they must implement the mount, init, destroy, forward backward functions.
//Furthermore they must and add any parameters to be optimized to the optParams list with the corresponding gradients added to the dOptParams list

struct layer {
    struct vecArr y;
    struct vecArr dy;
    void (*mount)(Layer l, struct vecArr x);
    void (*init)(Layer l, struct vecArr x);
    void (*destroy)(Layer l);
    void (*forward)(Layer l, struct vecArr x);
    void (*backward)(Layer l, struct vecArr x, struct vecArr dx);
    struct list optParams; //list of parameters to optimize
    struct list dOptParams; //list of gradients of parameters to optimize
    struct list optimizers; //corresponding optimizers
};

void LayerMount(Layer l, struct vecArr x); //mounting creates all temporary buffers
void LayerInit(Layer l, struct vecArr x); //initilalization is internal creation for a clean UI
void LayerDestroy(Layer l); //memory destruction
void LayerForward(Layer l, struct vecArr x); //forward pass
void LayerBackward(Layer l, struct vecArr x, struct vecArr dx); //backward pass, presumes previous call to LayerForward and that the model state reflects this

#endif //LAYER_H
