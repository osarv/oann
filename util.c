#include <stdlib.h>
#include <stdio.h>
#include "util.h"

void CheckPtr(void* ptr) {
    if (!ptr) exit(EXIT_FAILURE);
}
