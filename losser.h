#ifndef LOSSER_H
#define LOSSER_H
#include "oann.h"
#include "vecarr.h"

//everything is owned and freed by the network
struct losser {
    VecArr y;
    void (*forward)(Losser l, VecArr x);
    void (*backward)(Losser l, VecArr x, VecArr labels, VecArr dx);
    OANNfloat (*calcLoss)(Losser l, VecArr x, VecArr labels);
};

#endif //LOSSER_H
