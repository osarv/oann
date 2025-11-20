#include <stdlib.h>
#include <stdio.h>
#include "operation.h"

#ifndef TEST //no test
int main() {
}
#endif //TEST

#ifdef TEST //test
void TestOperationDenseForwardAndBackward();
int main() {
    TestOperationDenseForwardAndBackward();
}
#endif //TEST
