#ifndef DATASET_H
#define DATASET_H
#include "vecarr.h"

struct dataset {
    int trainBatchSize;
    int testBatchSize;
    int nTrainBatches;
    int nTestBatches;
    struct vecArr (*getTrainFeatures)(Dataset d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTrainLabels)(Dataset d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTestFeatures)(Dataset d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTestLabels)(Dataset d, struct vecArr buf, int batchIdx); //no idx checking
};

#endif //DATASET_H
