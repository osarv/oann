#ifndef DATASET_H
#define DATASET_H
#include "vecarr.h"

struct dataset {
    int nTrainSamples;
    int nTestSamples;
    void (*getTrainFeatures)(VecArr buf, int sampleIdxStart);
    void (*getTrainLabels)(VecArr buf, int sampleIdxStart);
    void (*getTestFeatures)(VecArr buf, int sampleIdxStart);
    void (*getTestLabels)(VecArr buf, int sampleIdxStart);
};

#endif //DATASET_H
