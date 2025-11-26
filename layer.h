#ifndef LAYER_H
#define LAYER_H
#include "vecarr.h"
#include "util.h"

//All layer implementations must include the layer struct, preferably as the first member.
//Additionally, they must implement the mount, init, destroy, forward backward functions.
//Furthermore they must and add any parameters to be optimized to the optParams list with the corresponding gradients added to the optParamGrads list

typedef struct layer* Layer;
struct layer {
    struct vecArr outF;
    struct vecArr inB;
    void (*mount)(Layer l, struct vecArr inF);
    void (*init)(Layer l, struct vecArr inF);
    void (*destroy)(Layer l);
    void (*forward)(Layer l, struct vecArr inF);
    void (*backward)(Layer l, struct vecArr inF, struct vecArr outB);
    struct list optParams; //list of parameters to optimize
    struct list optParamGrads; //list of gradients of parameters to optimize
    struct list optimizers; //corresponding optimizers
};

void LayerMount(Layer l, struct vecArr inF); //mounting creates all temporary buffers
void LayerInit(Layer l, struct vecArr inF); //initilalization is internal creation for a clean UI
void LayerDestroy(Layer l); //memory destruction
void LayerForward(Layer l, struct vecArr inF); //forward pass
void LayerBackward(Layer l, struct vecArr inF, struct vecArr outB); //backward pass, presumes previous call to LayerForward and that the model state reflects this

#endif //LAYER_H
