/* cuda.c - the one place C's types meet olang's for the GPU backend (docs/redesign.md 1.10, "The binding").
 *
 * Imported by kernels/gpu/device.olang as a C module (olang X17, X20): the CUDA headers are read by clang as olang
 * imports them, and these functions are compiled into the program. Everything crosses as plain numbers:
 *   - device addresses are uint64_t (U64 in olang, as the driver API's CUdeviceptr is), so a view at an offset is
 *     arithmetic in olang and the cast to the void * or float * a library takes is made here;
 *   - opaque handles (streams, events, cuBLAS and cuBLASLt handles) are uint64_t too;
 *   - host arrays cross as arrays (X3, X22: an olang Array passes as its storage's address), with an offset in bytes.
 *
 * The shim owns no state. A Device (device.olang) owns a status record, made here (oann_status_new) and freed by the
 * Device's destructor: the calls on the hot path (copies, products) return nothing and record the first status that
 * is not success in it, with the call's name, so an olang caller does not branch per call; Device.Sync gives that
 * first error back as a GpuError and clears the record. Setup calls (Open, Buf.Make) return their status directly.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <nvrtc.h>

/* ---- the status record: the first failure since it was last cleared ---- */

enum { OANN_OK = 0, OANN_CUDA = 1, OANN_CUBLAS = 2 };

typedef struct {
    int32_t kind;      /* OANN_OK, OANN_CUDA or OANN_CUBLAS */
    int32_t code;      /* the library's status */
    const char *what;  /* the call that failed: a literal of this file */
} oann_status;

uint64_t oann_status_new(void) { return (uint64_t)(uintptr_t)calloc(1, sizeof(oann_status)); }
void oann_status_free(uint64_t st) { free((void *)(uintptr_t)st); }
int32_t oann_status_kind(uint64_t st) { return ((oann_status *)(uintptr_t)st)->kind; }
int32_t oann_status_code(uint64_t st) { return ((oann_status *)(uintptr_t)st)->code; }
const char *oann_status_what(uint64_t st) { return ((oann_status *)(uintptr_t)st)->what; }
void oann_status_clear(uint64_t st) { memset((void *)(uintptr_t)st, 0, sizeof(oann_status)); }

static void note(uint64_t st, int32_t kind, int32_t code, const char *what) {
    oann_status *s = (oann_status *)(uintptr_t)st;
    if (code != 0 && s->kind == OANN_OK) {
        s->kind = kind;
        s->code = code;
        s->what = what;
    }
}

/* ---- the device ---- */

/* the number of devices; a failure as the negated cudaError_t */
int32_t oann_device_count(void) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    return e == cudaSuccess ? n : -(int32_t)e;
}

/* the device's name into name (NUL-ended, at most len bytes), its compute capability and memory into info:
 * major, minor, multiprocessors, then the global memory in MiB */
int32_t oann_device_info(int32_t ordinal, char *name, int32_t len, int32_t *info) {
    struct cudaDeviceProp p;
    cudaError_t e = cudaGetDeviceProperties(&p, ordinal);
    if (e != cudaSuccess) return e;
    strncpy(name, p.name, (size_t)len);
    if (len > 0) name[len - 1] = 0;
    info[0] = p.major;
    info[1] = p.minor;
    info[2] = p.multiProcessorCount;
    info[3] = (int32_t)(p.totalGlobalMem >> 20);
    return 0;
}

int32_t oann_set_device(int32_t ordinal) { return cudaSetDevice(ordinal); }

/* a stream that does not synchronize with the legacy default stream */
int32_t oann_stream_create(uint64_t *out) {
    cudaStream_t s = 0;
    cudaError_t e = cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    *out = (uint64_t)(uintptr_t)s;
    return e;
}

int32_t oann_stream_destroy(uint64_t s) { return cudaStreamDestroy((cudaStream_t)(uintptr_t)s); }

/* an event for ordering only (no timing) */
int32_t oann_event_create(uint64_t *out) {
    cudaEvent_t ev = 0;
    cudaError_t e = cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
    *out = (uint64_t)(uintptr_t)ev;
    return e;
}

int32_t oann_event_destroy(uint64_t ev) { return cudaEventDestroy((cudaEvent_t)(uintptr_t)ev); }

/* waits for everything enqueued on the stream; its status, which is also where asynchronous faults are reported */
int32_t oann_stream_sync(uint64_t s) { return cudaStreamSynchronize((cudaStream_t)(uintptr_t)s); }

/* ---- device memory ---- */

int32_t oann_malloc(uint64_t *out, uint64_t bytes) {
    void *p = 0;
    cudaError_t e = cudaMalloc(&p, (size_t)bytes);
    *out = (uint64_t)(uintptr_t)p;
    return e;
}

int32_t oann_free(uint64_t p) { return cudaFree((void *)(uintptr_t)p); }

/* bytes from host + from (bytes) to the device address dst, on the copy stream, then ordered before what is enqueued
 * next on the compute stream by an event. From pageable memory the call returns once host's bytes are staged, so the
 * host array may be written again at once */
void oann_upload(uint64_t st, uint64_t dst, const void *host, uint64_t from, uint64_t bytes, uint64_t copy, uint64_t event,
                 uint64_t compute) {
    cudaStream_t cs = (cudaStream_t)(uintptr_t)copy;
    note(st, OANN_CUDA, cudaMemcpyAsync((void *)(uintptr_t)dst, (const char *)host + from, (size_t)bytes,
                                        cudaMemcpyHostToDevice, cs), "cudaMemcpyAsync (upload)");
    note(st, OANN_CUDA, cudaEventRecord((cudaEvent_t)(uintptr_t)event, cs), "cudaEventRecord (upload)");
    note(st, OANN_CUDA, cudaStreamWaitEvent((cudaStream_t)(uintptr_t)compute, (cudaEvent_t)(uintptr_t)event, 0),
         "cudaStreamWaitEvent (upload)");
}

/* bytes from the device address src to host + to (bytes), on the compute stream, after everything enqueued on it */
void oann_download(uint64_t st, void *host, uint64_t to, uint64_t src, uint64_t bytes, uint64_t compute) {
    note(st, OANN_CUDA, cudaMemcpyAsync((char *)host + to, (const void *)(uintptr_t)src, (size_t)bytes,
                                        cudaMemcpyDeviceToHost, (cudaStream_t)(uintptr_t)compute), "cudaMemcpyAsync (download)");
}

/* ---- cuBLAS ---- */

/* a cuBLAS handle on the stream with a fixed workspace (set after the stream: cublasSetStream resets it to the
 * library's pool, which would allocate per call) */
int32_t oann_cublas_create(uint64_t *out, uint64_t stream, uint64_t work, uint64_t workBytes) {
    cublasHandle_t h = 0;
    cublasStatus_t s = cublasCreate(&h);
    *out = (uint64_t)(uintptr_t)h;
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    /* products in the storage's own precision only: no TF32, and no split-K reduction rounded to BF16 or F16 between
     * its parts (cuBLAS's default for half-precision storage, a 3% error at a depth of 257 against kernels/gemm) */
    s = cublasSetMathMode(h, (cublasMath_t)(CUBLAS_DEFAULT_MATH | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    s = cublasSetStream(h, (cudaStream_t)(uintptr_t)stream);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    return cublasSetWorkspace(h, (void *)(uintptr_t)work, (size_t)workBytes);
}

int32_t oann_cublas_destroy(uint64_t h) { return cublasDestroy((cublasHandle_t)(uintptr_t)h); }

int32_t oann_cublaslt_create(uint64_t *out) {
    cublasLtHandle_t h = 0;
    cublasStatus_t s = cublasLtCreate(&h);
    *out = (uint64_t)(uintptr_t)h;
    return s;
}

int32_t oann_cublaslt_destroy(uint64_t h) { return cublasLtDestroy((cublasLtHandle_t)(uintptr_t)h); }

/* C = alpha op(A) op(B) + beta C in cuBLAS's column-major terms (blas.olang exchanges the operands for row-major):
 * cublasGemmEx with the types given (cudaDataType values) and the compute type (cublasComputeType_t); alpha and beta
 * are given as doubles and passed as the compute type's scalars (float for CUBLAS_COMPUTE_32F, double for 64F) */
void oann_gemm(uint64_t st, uint64_t h, bool ta, bool tb, int32_t m, int32_t n, int32_t k, double alpha,
               uint64_t a, int32_t atype, int32_t lda, uint64_t b, int32_t btype, int32_t ldb, double beta,
               uint64_t c, int32_t ctype, int32_t ldc, int32_t compute) {
    float alpha32 = (float)alpha, beta32 = (float)beta;
    int wide = compute == CUBLAS_COMPUTE_64F || compute == CUBLAS_COMPUTE_64F_PEDANTIC;
    const void *pa = wide ? (const void *)&alpha : (const void *)&alpha32;
    const void *pb = wide ? (const void *)&beta : (const void *)&beta32;
    note(st, OANN_CUBLAS,
         cublasGemmEx((cublasHandle_t)(uintptr_t)h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n,
                      k, pa, (const void *)(uintptr_t)a, (cudaDataType)atype, lda, (const void *)(uintptr_t)b,
                      (cudaDataType)btype, ldb, pb, (void *)(uintptr_t)c, (cudaDataType)ctype, ldc,
                      (cublasComputeType_t)compute, CUBLAS_GEMM_DEFAULT),
         "cublasGemmEx");
}
