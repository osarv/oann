#ifndef OANN_H
#define OANN_H

typedef struct layer* Layer;
Layer DenseCreate(int nOut);
Layer ReluCreate();

#endif //OANN_H
