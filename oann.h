#ifndef OANN_H
#define OANN_H

typedef struct network* Network;
typedef struct layer* Layer;
typedef struct optimizer* Optimizer;
typedef struct losser* Losser;
Network NetworkCreate(int inputVecLen);
void NetworkDestroy(Network n);
void NetworkAddLayer(Network n, Layer l);
Layer DenseCreate(int nOut);
Layer ReluCreate();
Losser SCECreate();
Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay);

#endif //OANN_H
