#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>
#include "util.h"

void* MallocOrCrash(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) exit(EXIT_FAILURE);
    return ptr;
}

void* CallocOrCrash(size_t size) {
    void* ptr = calloc(size, 1);
    if (!ptr) exit(EXIT_FAILURE);
    return ptr;
}

void* ReallocOrCrash(void* oldPtr, size_t size) {
    void* ptr = realloc(oldPtr, size);
    if (!ptr) exit(EXIT_FAILURE);
    return ptr;
}

void ErrorAndCrash(char* errMsg) {
    fputs(COLOR_RED "ERROR: " COLOR_RESET, stderr);
    fputs(errMsg, stderr);
    fputs("\n", stderr);
    exit(EXIT_FAILURE);
}

void ErrorBugFound() {
    fputs(COLOR_RED "ERROR: bug found" COLOR_RESET, stderr);
    exit(EXIT_FAILURE);
}

bool CheckIfFileExists(char* path) {
    FILE* fp = fopen(path, "r");
    if (fp) {
        fclose(fp);
        return true;
    }
    return false;
}

bool DownloadFile(char* url, char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) return false;
    CURL* curl = curl_easy_init();
    if (!curl) {
        fclose(fp);
        return false;
    }
    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_HEADER, 0);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(fp);
    if (res != CURLE_OK) return false;
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    return status == 200;
}

bool DecompressGzFile(char* src, char* dst) {
    FILE* out = fopen(dst, "w");
    if (!out) return false;
    gzFile in = gzopen(src, "r");
    if (!in) {fclose(out); return false;}
    char buf[8192];
    int n;
    while ((n = gzread(in, buf, sizeof(buf))) > 0) {
        if (fwrite(buf, 1, n, out) != (unsigned long)n) {
            gzclose(in);
            fclose(out);
            return false;
        }
    }
    gzclose(in);
    fclose(out);
    if (n <= -1) return false;
    int errNum;
    gzerror(in, &errNum);
    return errNum == Z_OK;
}

struct list ListInit(int elemSize) {
    struct list l = (struct list){0};
    l.elemSize = elemSize;
    return l;
}

struct list ListSlice(struct list* l, int start, int end) { //list slices must not be added to
    struct list lSlice;
    lSlice.elemSize = l->elemSize;
    lSlice.len = end - start;
    lSlice.cap = 0;
    lSlice.ptr = (char*)l->ptr + start * l->elemSize;
    return lSlice;
}

void ListClear(struct list* l) {
    l->len = 0;
}

void ListDestroy(struct list l) {
    if (l.ptr) free(l.ptr);
}

void listCopyElem(void* ptr, void* elem, int elemSize) {
    for (int i = 0; i < elemSize; i++) {
        ((char*)ptr)[i] = ((char*)elem)[i];
    }
}

#define LIST_ALLOC_STEP_SIZE 100
void ListAdd(struct list* l, void* elem) {
    if (l->elemSize == 0) ErrorBugFound();
    if (l->len >= l->cap) {
        if (l->ptr && l->cap == 0) ErrorBugFound(); //tried to add to slice
        l->cap += LIST_ALLOC_STEP_SIZE;
        l->ptr = ReallocOrCrash(l->ptr, l->elemSize * l->cap);
    }
    memcpy((char*)l->ptr + l->len * l->elemSize, elem, l->elemSize);
    l->len++;
}

void ListAddList(struct list* head, struct list tail) {
    if (head->elemSize != tail.elemSize) ErrorBugFound();
    for (int i = 0; i < tail.len; i++) {
        ListAdd(head, &tail.ptr + i * tail.elemSize);
    }
}

void ListRetract(struct list* l, int newLen) {
    if (newLen > l->len) ErrorBugFound();
    l->len = newLen;
}

void* ListGetIdx(struct list* l, int idx) {
    if (idx >= l->len) ErrorBugFound();
    return (char*)l->ptr + idx * l->elemSize;
}

void* ListGetCmp(struct list* l, void* cmpVal, bool(*cmpFunc)(void* cmpVal, void* listElem)) {
    for (int i = 0; i < l->len; i++) {
        void* listElem = (char*)l->ptr + l->elemSize * i;
        if (cmpFunc(cmpVal, listElem)) return listElem;
    }
    return NULL;
}
