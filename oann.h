#ifndef OANN_H
#define OANN_H

typedef struct network* Network;
typedef struct layer* Layer;
typedef struct optimizer* Optimizer;
typedef struct losser* Losser;
typedef struct dataset* DataSet;
Layer DenseCreate(int nOut);
Layer ReluCreate();
Losser SCECreate();
Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay);
Network NetworkCreate(int inputVecLen, Optimizer o);
void NetworkDestroy(Network n);
void NetworkAddLayer(Network n, Layer l);
void NetworkTrain(Network n, DataSet d);
float NetworkTest(Network n, DataSet d);

#endif //OANN_H
