#ifndef UTIL_H
#define UTIL_H
#include <stdbool.h>

#define COLOR_RESET "\x1b[0m"
#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"


#ifdef TEST
#undef TEST
#define TEST(func) __attribute__((constructor)) static void Test##func()
#endif //TEST

#ifndef TEST
#undef TEST
#define TEST(func) __attribute__((unused)) static void Test##func()
#endif //TEST

#define TEST_PASSED {printf(COLOR_GREEN "%s passed\n" COLOR_RESET, __func__); return;}
#define TEST_FAILED {printf(COLOR_RED "%s failed\n" COLOR_RESET, __func__); return;}

//error handling
void ErrorBugFound();
void ErrorAndCrash(char* errMsg);
void* MallocOrCrash(size_t size);
void* CallocOrCrash(size_t size);
void* ReallocOrCrash(void* oldPtr, size_t size);
bool CheckIfFileExists(char* path);
bool DownloadFile(char* url, char* path);
bool DecompressGzFile(char* src, char* dst);

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
