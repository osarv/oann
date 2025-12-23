#ifndef DATASET_H
#define DATASET_H
#include "vecarr.h"

struct dataset {
    int nTrainSamples;
    int nTestSamples;
    int featureSize;
    int labelSize;
    void (*getTrainFeatures)(Dataset d, struct vecArr buf, int sampleIdxStart); //no idx checking
    void (*getTrainLabels)(Dataset d, struct vecArr buf, int sampleIdxStart); //no idx checking
    void (*getTestFeatures)(Dataset d, struct vecArr buf, int sampleIdxStart); //no idx checking
    void (*getTestLabels)(Dataset d, struct vecArr buf, int sampleIdxStart); //no idx checking
};

#endif //DATASET_H
