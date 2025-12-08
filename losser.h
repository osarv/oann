#ifndef LOSSER_H
#define LOSSER_H
#include "oann.h"
#include "vecarr.h"

struct losser {
    struct vecArr y;
    void (*forward)(Losser l, struct vecArr x);
    void (*backward)(Losser l, struct vecArr x, struct vecArr labels, struct vecArr dx);
    float (*calcLoss)(Losser l, struct vecArr x, struct vecArr labels);
};

#endif //LOSSER_H
