#ifndef OANN_H
#define OANN_H

typedef struct network* Network;
typedef struct layer* Layer;
typedef struct optimizer* Optimizer;
typedef struct losser* Losser;
typedef struct dataset* Dataset;
Layer DenseCreate(int nOut);
Layer ReluCreate();
Losser SCECreate();
Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay);
Network NetworkCreate(int nIn, Losser lsr, Optimizer o);
Dataset MnistCreate(int trainBatchSize, int testBatchSize);
void NetworkDestroy(Network n);
void NetworkAddLayer(Network n, Layer l);
float NetworkTrain(Network n, Dataset d);
float NetworkTest(Network n, Dataset d);

#endif //OANN_H
