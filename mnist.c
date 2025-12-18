#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "dataset.h"
#include "util.h"

#define MNIST_FEATURE_SIZE 784
#define MNIST_LABEL_SIZE 10
#define MNIST_N_TRAIN_SAMPLES 60000
#define MNIST_N_TEST_SAMPLES 10000
#define MNIST_PATH "dataset/mnist/"
#define MNIST_BASE_URL "https://storage.googleapis.com/cvdf-datasets/mnist/"
#define MNIST_TRAIN_FEATURES MNIST_BASE_URL "train-images-idx3-ubyte.gz"
#define MNIST_TRAIN_LABELS MNIST_BASE_URL "train-labels-idx1-ubyte.gz"
#define MNIST_TEST_FEATURES MNIST_BASE_URL "t10k-images-idx3-ubyte.gz"
#define MNIST_TEST_LABELS MNIST_BASE_URL "t10k-labels-idx1-ubyte.gz"

void mnistDownloadIfNeeded() {
    if (CheckIfFileExists(MNIST_PATH "trainf") &&
            CheckIfFileExists(MNIST_PATH "trainl") &&
            CheckIfFileExists(MNIST_PATH "testf") &&
            CheckIfFileExists(MNIST_PATH "testl")) return;
    else if (CheckIfFileExists(MNIST_PATH "trainf.gz") &&
            CheckIfFileExists(MNIST_PATH "trainl.gz") &&
            CheckIfFileExists(MNIST_PATH "testf.gz") &&
            CheckIfFileExists(MNIST_PATH "testl.gz"));
    else if (DownloadFile(MNIST_TRAIN_FEATURES, MNIST_PATH "trainf.gz") &&
            DownloadFile(MNIST_TRAIN_LABELS, MNIST_PATH "trainl.gz") &&
            DownloadFile(MNIST_TEST_FEATURES, MNIST_PATH "testf.gz") &&
            DownloadFile(MNIST_TEST_LABELS, MNIST_PATH "testl.gz"));
    else ErrorAndCrash("the MNIST dataset could not be downloaded");

    if (DecompressGzFile(MNIST_PATH "trainf.gz", MNIST_PATH "trainf") &&
            DecompressGzFile(MNIST_PATH "trainl.gz", MNIST_PATH "trainl") &&
            DecompressGzFile(MNIST_PATH "testf.gz", MNIST_PATH "testf") &&
            DecompressGzFile(MNIST_PATH "testl.gz", MNIST_PATH "testl"));
    else (ErrorAndCrash("the MNIST dataset could not be decompressed"));
}

Dataset MnistCreate(int trainBatchSize, int testBatchSize) {
    if (MNIST_N_TRAIN_SAMPLES % trainBatchSize != 0) ErrorAndCrash("mnist trainBatchSize must divide 60k");
    if (MNIST_N_TEST_SAMPLES % testBatchSize != 0) ErrorAndCrash("mnist testBatchSize must divide 10k");
    mnistDownloadIfNeeded();
    Dataset m = MallocOrCrash(sizeof(struct dataset));
    m->trainBatchSize = trainBatchSize;
    m->trainBatchSize = trainBatchSize;
    m->nTrainBatches = MNIST_N_TRAIN_SAMPLES / trainBatchSize;
    m->nTestBatches = MNIST_N_TEST_SAMPLES / testBatchSize;
    return m;
}
