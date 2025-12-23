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
float NetworkTrain(Network n, int batchSize, Dataset d);
float NetworkTest(Network n, int batchSize, Dataset d);
float NetworkInferTrainSample(Network n, Dataset d, int idx, float** features, float** predictions);

#endif //OANN_H
