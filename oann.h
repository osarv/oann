#ifndef OANN_H
#define OANN_H

#define MNIST_N_FEATURES 784
#define MNIST_N_LABELS 10

typedef struct network* Network;
typedef struct layer* Layer;
typedef struct optimizer* Optimizer;
typedef struct losser* Losser;
typedef struct dataset* Dataset;
Layer DenseCreate(int nOut);
Layer ReluCreate();
Losser SceCreate();
Optimizer AdamWCreate(float lr, float mDecay, float vDecay, float wDecay);
Network NetworkCreate(int nIn, Optimizer o);
Dataset MnistCreate();
void NetworkDestroy(Network n);
void NetworkAddLayer(Network n, Layer l);
void NetworkSetLosser(Network n, Losser lsr); //completes the network creation
float NetworkTrain(Network n, int batchSize, Dataset d);
float NetworkTest(Network n, int batchSize, Dataset d);
float NetworkInferTrainSample(Network n, Dataset d, int idx, float** features, float** predictions);

#endif //OANN_H
