#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"

typedef struct layer* Layer;

enum layerType {
    LAYER_HASHDENSE
};

void LayerDestroy(Layer l);
Layer HashDenseCreate(int nOut, int nVectors, int nBuckets, int rehashRate);

#endif //LAYER_H
