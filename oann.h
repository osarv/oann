#ifndef OANN_H
#define OANN_H

#define MNIST_N_FEATURES 784
#define MNIST_N_LABELS 10

typedef float OANNfloat;
typedef struct network* Network;
typedef struct layer* Layer;
typedef struct optimizer* Optimizer;
typedef struct losser* Losser;
typedef struct dataset* Dataset;
Layer DenseCreate(int nOut);
Layer ReluCreate();
Losser SceCreate();
Optimizer AdamWCreate(OANNfloat lr, OANNfloat mDecay, OANNfloat vDecay, OANNfloat wDecay);
Network NetworkCreate(int nIn, Optimizer o);
Dataset MnistCreate();
void NetworkDestroy(Network n);
void NetworkAddLayer(Network n, Layer l);
void NetworkSetLosser(Network n, Losser lsr); //completes the network creation
OANNfloat NetworkTrain(Network n, int batchSize, Dataset d);
OANNfloat NetworkTrainOneBatch(Network n, int batchSize, int sIdx, Dataset d);
OANNfloat NetworkTest(Network n, int batchSize, Dataset d);
OANNfloat NetworkInferTrainSample(Network n, Dataset d, int idx, OANNfloat** features, OANNfloat** predictions);

#endif //OANN_H
