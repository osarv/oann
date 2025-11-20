#ifndef UTIL_H
#define UTIL_H
#include <stdbool.h>

#define TEST_PASSED {printf("\x1b[32m" "Test Passed\n" "\x1b[0m"); return;}
#define TEST_FAILED {printf("\x1b[31m" "Test failed\n" "\x1b[0m"); return;}

//error handling
void ErrorBugFound();
void* MallocOrCrash(size_t size);
void* ReallocOrCrash(void* oldPtr, size_t size);

//lists
struct list {
    int elemSize;
    int len;
    int cap;
    void* ptr;
};

struct list ListInit(int elemSize);
struct list ListSlice(struct list* l, int start, int end); //list slices must not be added to
void ListClear(struct list* l);
void ListDestroy(struct list l);
void ListAdd(struct list* l, void* elem);
void ListAddList(struct list* head, struct list tail);
void ListRetract(struct list* l, int newLen);
void* ListGetIdx(struct list* l, int idx);
void* ListGetCmp(struct list* l, void* cmpVal, bool(*cmpFunc)(void* cmpVal, void* listElem));

#endif //UTIL_H
