#include <stdlib.h>
#include <stdio.h>
#include "oann.h"
#include "dataset.h"
#include "util.h"

#define MNIST_HEADER_SIZE_FEATURES 16
#define MNIST_HEADER_SIZE_LABELS 8
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

static void mnistGetTrainFeatures(VecArr buf, int sampleIdxStart) {
    FILE* fp = fopen(MNIST_PATH "trainf", "r");
    if (!fp) ErrorAndCrash("could not open mnist train features");
    if (fseek(fp, MNIST_HEADER_SIZE_FEATURES + sampleIdxStart * MNIST_N_FEATURES, SEEK_SET)) {
        ErrorAndCrash("could not seek in mnist train features");
    }
    if (fread(buf, 1, MNIST_N_FEATURES * VecArrNVecs(buf), fp) != (unsigned int)MNIST_N_FEATURES * VecArrNVecs(buf)) {
        ErrorAndCrash("could not read from mnist train features");
    }
    fclose(fp);
}

static void mnistGetTrainLabels(VecArr buf, int sampleIdxStart) {
    FILE* fp = fopen(MNIST_PATH "trainl", "r");
    if (!fp) ErrorAndCrash("could not open mnist train labels");
    if (fseek(fp, MNIST_HEADER_SIZE_LABELS + sampleIdxStart, SEEK_SET)) {
        ErrorAndCrash("could not seek in mnist train labels");
    }
    char labels[VecArrNVecs(buf)];
    if (fread(labels, 1, VecArrNVecs(buf), fp) != (unsigned int)VecArrNVecs(buf)) {
        ErrorAndCrash("could not read from mnist train labels");
    }
    VecArrInitConst(buf, 0);
    for (int i = 0; i < VecArrNVecs(buf); i++) buf[i * VecArrVecLen(buf) + labels[i]] = 1;
    fclose(fp);
}

static void mnistGetTestFeatures(VecArr buf, int sampleIdxStart) {
    FILE* fp = fopen(MNIST_PATH "testf", "r");
    if (!fp) ErrorAndCrash("could not open mnist test features");
    if (fseek(fp, MNIST_HEADER_SIZE_FEATURES + sampleIdxStart * MNIST_N_FEATURES, SEEK_SET)) {
        ErrorAndCrash("could not seek in mnist test features");
    }
    if (fread(buf, 1, MNIST_N_FEATURES * VecArrNVecs(buf), fp) != (unsigned int)MNIST_N_FEATURES * VecArrNVecs(buf)) {
        ErrorAndCrash("could not read from mnist test features");
    }
    fclose(fp);
}

static void mnistGetTestLabels(VecArr buf, int sampleIdxStart) {
    FILE* fp = fopen(MNIST_PATH "testl", "r");
    if (!fp) ErrorAndCrash("could not open mnist test labels");
    if (fseek(fp, MNIST_HEADER_SIZE_LABELS + sampleIdxStart, SEEK_SET)) {
        ErrorAndCrash("could not seek in mnist test labels");
    }
    char labels[VecArrNVecs(buf)];
    if (fread(labels, 1, VecArrNVecs(buf), fp) != (unsigned int)VecArrNVecs(buf)) {
        ErrorAndCrash("could not read from mnist test labels");
    }
    VecArrInitConst(buf, 0);
    for (int i = 0; i < VecArrNVecs(buf); i++) buf[i * VecArrVecLen(buf) + labels[i]] = 1;
    fclose(fp);
}

Dataset MnistCreate() {
    mnistDownloadIfNeeded();
    Dataset m = MallocOrCrash(sizeof(struct dataset));
    m->nTrainSamples = MNIST_N_TRAIN_SAMPLES;
    m->nTestSamples = MNIST_N_TEST_SAMPLES;
    m->getTrainFeatures = mnistGetTrainFeatures;
    m->getTrainLabels = mnistGetTrainLabels;
    m->getTestFeatures = mnistGetTestFeatures;
    m->getTestLabels = mnistGetTestLabels;
    return m;
}
