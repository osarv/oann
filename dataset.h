#ifndef DATASET_H
#define DATASET_H
#include "vecarr.h"

struct dataset {
    int trainBatchSize;
    int testBatchSize;
    int nTrainBatches;
    int nTestBatches;
    struct vecArr (*getTrainFeatures)(DataSet d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTrainLabels)(DataSet d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTestFeatures)(DataSet d, struct vecArr buf, int batchIdx); //no idx checking
    struct vecArr (*getTestLabels)(DataSet d, struct vecArr buf, int batchIdx); //no idx checking
};

#endif //DATASET_H
