#ifndef DATASET_H
#define DATASET_H
#include "vecarr.h"

struct dataset {
    int nBatches;
    int nTestBatches;
    struct vecArr (*feedFeatures)(DataSet d);
    struct vecArr (*feedTrainLabels)(DataSet d);
    struct vecArr (*feedTestFeatures)(DataSet d);
    struct vecArr (*feedTestLabels)(DataSet d);
};

#endif //DATASET_H
