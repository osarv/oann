#ifndef LAYER_H
#define LAYER_H
#include "oann.h"
#include "vecarr.h"
#include "util.h"

enum layerType {
    LAYER_MUL,
    LAYER_ADD,
    LAYER_RELU,
};

//everything is owned and freed by the network
struct layer {
    enum layerType type;
    OANNfloat varScaling;
    int nOut; //may be set in create or init
    void (*init)(Layer l, int nIn, OANNfloat varScalingNextLayer);
    VecArr (*forward)(Layer l, VecArr x); //returns y; y may be x
    VecArr (*backward)(Layer l, VecArr x, VecArr dx); //returns dx, dx may be dy; does NOT call the optimizers
    struct ptrList p; //list of parameters to optimize
    
    VecArr y; //OBS: created and filled by network
    VecArr dy; //OBS: created and filled by network
    struct ptrList dp; //OBS: created and filled by network; list of gradients of parameters to optimize
    struct ptrList optimizers; //OBS: created and filled by network; corresponding optimizer
};

#endif //LAYER_H
