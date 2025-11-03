#include <stdlib.h>
#include <stdio.h>
#include "util.h"

void* MallocOrCrash(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) exit(EXIT_FAILURE);
    return ptr;
}
