#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <curl/curl.h>
#include <zlib.h>
#include "layer.h"
#include "util.h"

#ifdef TEST
void main(){}

struct testArrs TestPrepareLayer(Layer l) {
    struct testArrs arrs;
    arrs.x = VecArrCreate(3, 2);
    arrs.dx = VecArrCreateSameDim(arrs.x);
    l->init(l, VecArrVecLen(arrs.x), 1);
    l->y = VecArrCreate(2, 2);
    l->dy = VecArrCreateSameDim(l->y);
    l->dp = PtrListInit();
    for (int i = 0; i < PtrListLen(l->p); i++) {
        VecArr p = PtrListGetIdx(l->p, i);
        VecArr dp = VecArrCreateSameDim(p);
        PtrListAdd(&(arrs.dp), dp);
    }
    return arrs;
}
#endif //TEST

void* MallocOrCrash(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fputs(COLOR_RED "ERROR: " COLOR_RESET "memory allocation failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* CallocOrCrash(size_t size) {
    void* ptr = calloc(size, 1);
    if (!ptr) {
        fputs(COLOR_RED "ERROR: " COLOR_RESET "memory allocation failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void* ReallocOrCrash(void* oldPtr, size_t size) {
    void* ptr = realloc(oldPtr, size);
    if (!ptr) {
        fputs(COLOR_RED "ERROR: " COLOR_RESET "memory allocation failed\n", stderr);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void ErrorAndCrash(char* errMsg) {
    fputs(COLOR_RED "ERROR: " COLOR_RESET, stderr);
    fputs(errMsg, stderr);
    fputs("\n", stderr);
    exit(EXIT_FAILURE);
}

void ErrorBugFound() {
    fputs(COLOR_RED "ERROR: bug found\n" COLOR_RESET, stderr);
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

struct ptrList PtrListInit() {
    struct ptrList l = (struct ptrList){0};
    return l;
}

void PtrListDestroy(struct ptrList l) {
    if (l.ptr) free(l.ptr);
}

#define PTR_LIST_ALLOC_STEP_SIZE 100
void PtrListAdd(struct ptrList* l, void* ptr) {
    if (l->len >= l->cap) {
        l->cap += PTR_LIST_ALLOC_STEP_SIZE;
        l->ptr = ReallocOrCrash(l->ptr, sizeof(void*) * l->cap);
    }
    l->ptr[l->len] = ptr;
    l->len++;
}

void* PtrListGetIdx(struct ptrList l, int idx) {
    if (idx >= l.len) ErrorBugFound();
    return l.ptr[idx];
}
