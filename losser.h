#ifndef LOSSER_H
#define LOSSER_H
#include "oann.h"
#include "vecarr.h"

struct losser {
    VecArr y;
    void (*unmount)(Losser l);
    void (*mount)(Losser l, VecArr x);
    void (*destroy)(Losser l);
    void (*forward)(Losser l, VecArr x);
    void (*backward)(Losser l, VecArr x, VecArr labels, VecArr dx);
    float (*calcLoss)(Losser l, VecArr x, VecArr labels);
};

#endif //LOSSER_H
