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
 * Device's destructor: the calls on the hot path (copies, products, launches) return nothing and record the first
 * status that is not success in it, with the call's name, so an olang caller does not branch per call; Device.Sync
 * gives that first error back as a GpuError and clears the record. Setup calls (Open, Buf.Make, a compile) return
 * their status directly.
 *
 * Our own kernels (kernels/gpu/kern.olang): NVRTC compiles CUDA C text to PTX; the driver API loads and launches it,
 * reached through cudaGetDriverEntryPoint into a table the caller owns (oann_driver_new), so nothing links libcuda.
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

enum { OANN_OK = 0, OANN_CUDA = 1, OANN_CUBLAS = 2, OANN_DRIVER = 3 };

typedef struct {
    int32_t kind;      /* OANN_OK, OANN_CUDA, OANN_CUBLAS or OANN_DRIVER */
    int32_t code;      /* the library's status */
    const char *what;  /* the call that failed: a literal of this file */
    const char *text;  /* the driver's own words for a driver status (cuGetErrorString's, static), else 0 */
} oann_status;

uint64_t oann_status_new(void) { return (uint64_t)(uintptr_t)calloc(1, sizeof(oann_status)); }
void oann_status_free(uint64_t st) { free((void *)(uintptr_t)st); }
int32_t oann_status_kind(uint64_t st) { return ((oann_status *)(uintptr_t)st)->kind; }
int32_t oann_status_code(uint64_t st) { return ((oann_status *)(uintptr_t)st)->code; }
const char *oann_status_what(uint64_t st) { return ((oann_status *)(uintptr_t)st)->what; }
const char *oann_status_text(uint64_t st) { return ((oann_status *)(uintptr_t)st)->text; }
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

/* ---- device memory, filled and copied on the compute stream (kernels/gpu/kern.olang) ---- */

/* bytes at dst set to value (a byte), on the stream, in order with everything else on it */
void oann_memset(uint64_t st, uint64_t dst, int32_t value, uint64_t bytes, uint64_t stream) {
    note(st, OANN_CUDA, cudaMemsetAsync((void *)(uintptr_t)dst, value, (size_t)bytes, (cudaStream_t)(uintptr_t)stream),
         "cudaMemsetAsync");
}

/* bytes from the device address src to dst, on the stream */
void oann_copy(uint64_t st, uint64_t dst, uint64_t src, uint64_t bytes, uint64_t stream) {
    note(st, OANN_CUDA, cudaMemcpyAsync((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)bytes,
                                        cudaMemcpyDeviceToDevice, (cudaStream_t)(uintptr_t)stream), "cudaMemcpyAsync (device)");
}

/* ---- timing: events that record time (kern.olang's Timer, for the benches) ---- */

int32_t oann_timer_create(uint64_t *out) {
    cudaEvent_t ev = 0;
    cudaError_t e = cudaEventCreate(&ev);
    *out = (uint64_t)(uintptr_t)ev;
    return e;
}

void oann_event_record(uint64_t st, uint64_t ev, uint64_t stream) {
    note(st, OANN_CUDA, cudaEventRecord((cudaEvent_t)(uintptr_t)ev, (cudaStream_t)(uintptr_t)stream), "cudaEventRecord");
}

/* the milliseconds from event a to event b, after waiting for b */
int32_t oann_event_ms(uint64_t a, uint64_t b, float *ms) {
    cudaError_t e = cudaEventSynchronize((cudaEvent_t)(uintptr_t)b);
    if (e != cudaSuccess) return e;
    return cudaEventElapsedTime(ms, (cudaEvent_t)(uintptr_t)a, (cudaEvent_t)(uintptr_t)b);
}

/* ---- NVRTC: CUDA C text to PTX (kern.olang compiles at plan time and caches the PTX by its text's hash) ---- */

/* NVRTC's version, major * 1000 + minor * 10 (as CUDA's are written), or the negated status */
int32_t oann_nvrtc_version(void) {
    int major = 0, minor = 0;
    nvrtcResult r = nvrtcVersion(&major, &minor);
    return r == NVRTC_SUCCESS ? major * 1000 + minor * 10 : -(int32_t)r;
}

/* compiles src (NUL-ended) with the options given one a line (NUL-ended, at most 32); the program in *prog whether
 * or not it compiled, since its log is read from it - destroyed by oann_nvrtc_destroy. NVRTC's status */
int32_t oann_nvrtc_compile(const char *src, const char *options, uint64_t *prog) {
    nvrtcProgram p = 0;
    nvrtcResult r = nvrtcCreateProgram(&p, src, "oann.cu", 0, 0, 0);
    *prog = (uint64_t)(uintptr_t)p;
    if (r != NVRTC_SUCCESS) return r;
    char buf[2048];
    const char *opts[32];
    int n = 0;
    size_t len = strlen(options);
    if (len >= sizeof buf) return NVRTC_ERROR_INVALID_OPTION;
    memcpy(buf, options, len + 1);
    for (char *s = buf; *s != 0 && n < 32;) {
        opts[n++] = s;
        char *e = strchr(s, '\n');
        if (e == 0) break;
        *e = 0;
        s = e + 1;
    }
    return nvrtcCompileProgram(p, n, opts);
}

/* the size in bytes of the program's log and of its PTX, each with its NUL */
uint64_t oann_nvrtc_log_size(uint64_t prog) {
    size_t n = 0;
    if (nvrtcGetProgramLogSize((nvrtcProgram)(uintptr_t)prog, &n) != NVRTC_SUCCESS) return 0;
    return n;
}

uint64_t oann_nvrtc_ptx_size(uint64_t prog) {
    size_t n = 0;
    if (nvrtcGetPTXSize((nvrtcProgram)(uintptr_t)prog, &n) != NVRTC_SUCCESS) return 0;
    return n;
}

/* the log and the PTX into out, which holds the size above */
int32_t oann_nvrtc_log(uint64_t prog, char *out) { return nvrtcGetProgramLog((nvrtcProgram)(uintptr_t)prog, out); }
int32_t oann_nvrtc_ptx(uint64_t prog, char *out) { return nvrtcGetPTX((nvrtcProgram)(uintptr_t)prog, out); }

void oann_nvrtc_destroy(uint64_t prog) {
    nvrtcProgram p = (nvrtcProgram)(uintptr_t)prog;
    if (p != 0) nvrtcDestroyProgram(&p);
}

/* ---- the driver API, for loading PTX and launching it ----
 *
 * Reached through the runtime (cudaGetDriverEntryPoint), so nothing links libcuda by a path: the entry points are a
 * table made by oann_driver_new and held by whoever loads a module (a Kernel in kern.olang), freed with it. The
 * driver's types are given here by their shapes (CUresult an int, handles pointers), so cuda.h is not imported. */

typedef int (*oann_cuModuleLoadData)(void **module, const void *image);
typedef int (*oann_cuModuleGetFunction)(void **f, void *module, const char *name);
typedef int (*oann_cuModuleUnload)(void *module);
typedef int (*oann_cuLaunchKernel)(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by,
                                   unsigned bz, unsigned shared, void *stream, void **params, void **extra);
typedef int (*oann_cuGetErrorString)(int code, const char **text);

typedef struct {
    oann_cuModuleLoadData load;
    oann_cuModuleGetFunction get;
    oann_cuModuleUnload unload;
    oann_cuLaunchKernel launch;
    oann_cuGetErrorString text;
} oann_driver;

static cudaError_t entry(const char *name, void **out) {
    enum cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSuccess;
    cudaError_t e = cudaGetDriverEntryPoint(name, out, cudaEnableDefault, &found);
    if (e == cudaSuccess && (found != cudaDriverEntryPointSuccess || *out == 0)) return cudaErrorSymbolNotFound;
    return e;
}

/* the table of the driver's entry points into *out (0 on failure); the runtime's status */
int32_t oann_driver_new(uint64_t *out) {
    *out = 0;
    oann_driver *d = (oann_driver *)calloc(1, sizeof(oann_driver));
    if (d == 0) return cudaErrorMemoryAllocation;
    cudaError_t e;
    if ((e = entry("cuModuleLoadData", (void **)&d->load)) != cudaSuccess ||
        (e = entry("cuModuleGetFunction", (void **)&d->get)) != cudaSuccess ||
        (e = entry("cuModuleUnload", (void **)&d->unload)) != cudaSuccess ||
        (e = entry("cuLaunchKernel", (void **)&d->launch)) != cudaSuccess ||
        (e = entry("cuGetErrorString", (void **)&d->text)) != cudaSuccess) {
        free(d);
        return e;
    }
    *out = (uint64_t)(uintptr_t)d;
    return cudaSuccess;
}

void oann_driver_free(uint64_t drv) { free((void *)(uintptr_t)drv); }

#define DRV(drv) ((oann_driver *)(uintptr_t)(drv))

/* the driver's text for its status code (static), "" where it has none */
const char *oann_driver_error(uint64_t drv, int32_t code) {
    const char *t = 0;
    if (DRV(drv)->text(code, &t) != 0 || t == 0) return "";
    return t;
}

/* image (PTX, NUL-ended) loaded into the current context - the device's, made current by Device.Open - as a module;
 * the driver's status */
int32_t oann_module_load(uint64_t drv, const void *image, uint64_t *module) {
    void *m = 0;
    int r = DRV(drv)->load(&m, image);
    *module = (uint64_t)(uintptr_t)m;
    return r;
}

int32_t oann_module_function(uint64_t drv, uint64_t module, const char *name, uint64_t *f) {
    void *p = 0;
    int r = DRV(drv)->get(&p, (void *)(uintptr_t)module, name);
    *f = (uint64_t)(uintptr_t)p;
    return r;
}

int32_t oann_module_unload(uint64_t drv, uint64_t module) { return DRV(drv)->unload((void *)(uintptr_t)module); }

/* the kernel f launched on the stream over grid x block threads with shared bytes of dynamic shared memory. Its
 * arguments are n slots of 64 bits (at most 32): a device address, or a number's bits in the slot's low bytes - the
 * driver copies each parameter's own size from the start of its slot, so a 32-bit parameter reads the low half
 * (little-endian). A failure is recorded in the status record with the driver's text */
void oann_launch(uint64_t st, uint64_t drv, uint64_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by,
                 uint32_t bz, uint32_t shared, uint64_t stream, const uint64_t *args, int32_t n) {
    void *params[32];
    if (n > 32) n = 32;
    for (int i = 0; i < n; i++) params[i] = (void *)&args[i];
    int r = DRV(drv)->launch((void *)(uintptr_t)f, gx, gy, gz, bx, by, bz, shared, (void *)(uintptr_t)stream, params, 0);
    oann_status *s = (oann_status *)(uintptr_t)st;
    if (r != 0 && s->kind == OANN_OK) {
        note(st, OANN_DRIVER, r, "cuLaunchKernel");
        s->text = oann_driver_error(drv, r);
    }
}

/* ==== cuBLASLt (GP3): products planned once and run many times ====
 *
 * A plan is a matmul descriptor, the three matrices' layouts and the algorithm cuBLASLt's heuristic chose for them,
 * made once (blas.olang's Product.Plan) and run as often as wanted: a run makes nothing and allocates nothing - its
 * workspace is the Device's, fixed at Open and shared with the cuBLAS handle (one stream, so never at once). Only
 * algorithms without a split-K reduction are taken (CUBLASLT_REDUCTION_SCHEME_NONE): every element is one sum in one
 * order, so a product run twice is the same bit for bit and no partial sum is rounded to a narrow output type.
 *
 * An algorithm is chosen for the alignment of the addresses it will read and write, which the heuristic cannot see:
 * the plan takes the addresses it is made with, keeps the alignment they share (at most 256 bytes) and tells the
 * heuristic so; a run whose addresses are less aligned than that is refused and recorded, never run. */

typedef struct {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatmulAlgo_t algo;
    cublasLtMatmulHeuristicResult_t offered[8];  /* the heuristic's, best first: what Tune times */
    int32_t found;
    int32_t compute;  /* cublasComputeType_t: the scalars' type */
    uint64_t align;   /* bytes every address of a run must be a multiple of */
    uint64_t work;    /* the workspace bytes the algorithm takes */
} oann_lt;

/* the largest power of two dividing x, at most 256 (x = 0: 256) */
static uint64_t lowbit(uint64_t x) {
    uint64_t b = x & (~x + 1);
    return b == 0 || b > 256 ? 256 : b;
}

static void lt_free(oann_lt *p) {
    if (!p) return;
    if (p->c) cublasLtMatrixLayoutDestroy(p->c);
    if (p->b) cublasLtMatrixLayoutDestroy(p->b);
    if (p->a) cublasLtMatrixLayoutDestroy(p->a);
    if (p->op) cublasLtMatmulDescDestroy(p->op);
    free(p);
}

void oann_lt_free(uint64_t plan) { lt_free((oann_lt *)(uintptr_t)plan); }

/* takes the heuristic's i-th algorithm and says what it is in info: its id, tile, split-K parts, reduction scheme, the
 * workspace it takes, how many were offered, the alignment, the ns a product took (0: not timed) and which it is */
static void choose(oann_lt *p, int i, int64_t *info, int64_t ns) {
    p->algo = p->offered[i].algo;
    p->work = p->offered[i].workspaceSize;
    int32_t v[4] = {0, 0, 0, 0};
    size_t got = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_ID, &v[0], sizeof v[0], &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v[1], sizeof v[1], &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v[2], sizeof v[2], &got);
    cublasLtMatmulAlgoConfigGetAttribute(&p->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &v[3], sizeof v[3], &got);
    for (int j = 0; j < 4; j++) info[j] = v[j];
    info[4] = (int64_t)p->work;
    info[5] = p->found;
    info[6] = (int64_t)p->align;
    info[7] = ns;
    info[8] = i;
}

static cublasStatus_t layout(cublasLtMatrixLayout_t *l, int32_t type, int32_t rows, int32_t cols, int64_t ld,
                             int32_t count, int64_t stride) {
    cublasStatus_t s = cublasLtMatrixLayoutCreate(l, (cudaDataType)type, (uint64_t)rows, (uint64_t)cols, ld);
    if (s != CUBLAS_STATUS_SUCCESS || count <= 1) return s;
    s = cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof count);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
    return cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof stride);
}

/* a plan for D = alpha op(A) op(B) + beta C in cuBLASLt's column-major terms, D in C's place (m x n from a depth of k),
 * each matrix with its cudaDataType and ld; count > 1 makes it count products, matrix i of each at i times its stride
 * (elements). reduction is the mask of split-K reduction schemes allowed (CUBLASLT_REDUCTION_SCHEME_NONE: none, so no
 * split-K; CUBLASLT_REDUCTION_SCHEME_MASK: any, partial sums possibly rounded to C's type). epilogue is a cublasLtEpilogue_t, bias its vector's address (0 none: then it must not need one). a, b,
 * c and bias are the addresses the plan is made with, align0 the alignment (bytes) some other address of a run must
 * share (an outer stride, 0 none): the algorithm is chosen for the alignment they all share. info gives what was
 * chosen: the algorithm's id, tile, split-K parts, reduction scheme, the workspace it takes, how many algorithms the
 * heuristic offered and the alignment. CUBLAS_STATUS_NOT_SUPPORTED where none fits */
int32_t oann_lt_plan(uint64_t *out, uint64_t lt, int32_t compute, int32_t scale, bool ta, bool tb, int32_t m,
                     int32_t n, int32_t k, int32_t atype, int64_t lda, int64_t sa, uint64_t a, int32_t btype,
                     int64_t ldb, int64_t sb, uint64_t b, int32_t ctype, int64_t ldc, int64_t sc, uint64_t c,
                     int32_t count, int32_t reduction, int32_t epilogue, uint64_t bias, uint64_t align0,
                     uint64_t workBytes, int64_t *info) {
    *out = 0;
    oann_lt *p = (oann_lt *)calloc(1, sizeof(oann_lt));
    if (!p) return CUBLAS_STATUS_ALLOC_FAILED;
    p->compute = compute;
    cublasStatus_t s = cublasLtMatmulDescCreate(&p->op, (cublasComputeType_t)compute, (cudaDataType)scale);
    cublasOperation_t oa = ta ? CUBLAS_OP_T : CUBLAS_OP_N, ob = tb ? CUBLAS_OP_T : CUBLAS_OP_N;
    uint32_t epi = (uint32_t)epilogue;
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &oa, sizeof oa);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSB, &ob, sizeof ob);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi);
    if (s == CUBLAS_STATUS_SUCCESS && bias != 0) {
        const void *pb = (const void *)(uintptr_t)bias;
        s = cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &pb, sizeof pb);
    }
    if (s == CUBLAS_STATUS_SUCCESS)
        s = layout(&p->a, atype, ta ? k : m, ta ? m : k, lda, count, sa);
    if (s == CUBLAS_STATUS_SUCCESS)
        s = layout(&p->b, btype, tb ? n : k, tb ? k : n, ldb, count, sb);
    if (s == CUBLAS_STATUS_SUCCESS) s = layout(&p->c, ctype, m, n, ldc, count, sc);
    uint64_t al = lowbit(a) < lowbit(b) ? lowbit(a) : lowbit(b);
    if (lowbit(c) < al) al = lowbit(c);
    if (bias != 0 && lowbit(bias) < al) al = lowbit(bias);
    if (align0 != 0 && lowbit(align0) < al) al = lowbit(align0);
    p->align = al;
    cublasLtMatmulPreference_t pref = 0;
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatmulPreferenceCreate(&pref);
    uint32_t red = (uint32_t)reduction;
    uint32_t al32 = (uint32_t)al;
    if (s == CUBLAS_STATUS_SUCCESS)
        s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workBytes, sizeof workBytes);
    if (s == CUBLAS_STATUS_SUCCESS)
        s = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &red, sizeof red);
    cublasLtMatmulPreferenceAttributes_t aligns[4] = {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                      CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    for (int i = 0; i < 4 && s == CUBLAS_STATUS_SUCCESS; i++)
        s = cublasLtMatmulPreferenceSetAttribute(pref, aligns[i], &al32, sizeof al32);
    cublasLtMatmulHeuristicResult_t *h = p->offered;
    int found = 0;
    if (s == CUBLAS_STATUS_SUCCESS)
        s = cublasLtMatmulAlgoGetHeuristic((cublasLtHandle_t)(uintptr_t)lt, p->op, p->a, p->b, p->c, p->c, pref, 8, h, &found);
    p->found = found;
    if (pref) cublasLtMatmulPreferenceDestroy(pref);
    if (s == CUBLAS_STATUS_SUCCESS && found == 0) s = CUBLAS_STATUS_NOT_SUPPORTED;
    if (s != CUBLAS_STATUS_SUCCESS) {
        lt_free(p);
        return s;
    }
    *out = (uint64_t)(uintptr_t)p;
    choose(p, 0, info, 0);
    return CUBLAS_STATUS_SUCCESS;
}

/* runs a plan on these addresses, outer times: run i at a + i oa, b + i ob, c + i oc (bytes) - a batch whose matrices
 * take two strides is outer runs of the plan's count. alpha and beta as the plan's scalars (float for 32F compute,
 * double for 64F, int32_t for 32I). An address less aligned than the plan's is recorded and nothing runs */
void oann_lt_run(uint64_t st, uint64_t lt, uint64_t plan, double alpha, double beta, uint64_t a, uint64_t b, uint64_t c,
                 int32_t outer, int64_t oa, int64_t ob, int64_t oc, uint64_t work, uint64_t workBytes, uint64_t stream) {
    oann_lt *p = (oann_lt *)(uintptr_t)plan;
    uint64_t mask = p->align - 1;
    if (((a | b | c) & mask) != 0 || (outer > 1 && (((uint64_t)oa | (uint64_t)ob | (uint64_t)oc) & mask) != 0)) {
        note(st, OANN_CUBLAS, CUBLAS_STATUS_NOT_SUPPORTED, "cublasLtMatmul: an address less aligned than its plan's");
        return;
    }
    float a32 = (float)alpha, b32 = (float)beta;
    int32_t ai = (int32_t)alpha, bi = (int32_t)beta;
    int wide = p->compute == CUBLAS_COMPUTE_64F || p->compute == CUBLAS_COMPUTE_64F_PEDANTIC;
    int integer = p->compute == CUBLAS_COMPUTE_32I || p->compute == CUBLAS_COMPUTE_32I_PEDANTIC;
    const void *pa = wide ? (const void *)&alpha : integer ? (const void *)&ai : (const void *)&a32;
    const void *pb = wide ? (const void *)&beta : integer ? (const void *)&bi : (const void *)&b32;
    for (int32_t i = 0; i < outer; i++) {
        void *ci = (void *)(uintptr_t)(c + (uint64_t)(i * oc));
        note(st, OANN_CUBLAS,
             cublasLtMatmul((cublasLtHandle_t)(uintptr_t)lt, p->op, pa, (const void *)(uintptr_t)(a + (uint64_t)(i * oa)), p->a,
                            (const void *)(uintptr_t)(b + (uint64_t)(i * ob)), p->b, pb, ci, p->c, ci, p->c, &p->algo,
                            (void *)(uintptr_t)work, (size_t)workBytes, (cudaStream_t)(uintptr_t)stream),
             "cublasLtMatmul");
    }
}

/* times every algorithm the heuristic offered on these addresses (as a run's, alpha 1, beta 0: c is overwritten), reps
 * products each after as many untimed, by events on the stream, and keeps the fastest; info as the plan's, and in
 * tried five numbers for each offered (8 at most): the ns a product took (-1: it failed or wants more workspace), its
 * algorithm's id, tile, split-K parts and reduction scheme. The status: the first failure, cuBLAS's or CUDA's (negated) */
int32_t oann_lt_tune(uint64_t lt, uint64_t plan, uint64_t a, uint64_t b, uint64_t c, int32_t outer, int64_t oa,
                     int64_t ob, int64_t oc, uint64_t work, uint64_t workBytes, uint64_t stream, int32_t reps,
                     int64_t *info, int64_t *tried) {
    oann_lt *p = (oann_lt *)(uintptr_t)plan;
    cudaStream_t s = (cudaStream_t)(uintptr_t)stream;
    cudaEvent_t e0 = 0, e1 = 0;
    cudaError_t ce = cudaEventCreate(&e0);
    if (ce == cudaSuccess) ce = cudaEventCreate(&e1);
    float one = 1, zero = 0;
    double one64 = 1, zero64 = 0;
    int32_t onei = 1, zeroi = 0;
    int wide = p->compute == CUBLAS_COMPUTE_64F || p->compute == CUBLAS_COMPUTE_64F_PEDANTIC;
    int integer = p->compute == CUBLAS_COMPUTE_32I || p->compute == CUBLAS_COMPUTE_32I_PEDANTIC;
    const void *pa = wide ? (const void *)&one64 : integer ? (const void *)&onei : (const void *)&one;
    const void *pz = wide ? (const void *)&zero64 : integer ? (const void *)&zeroi : (const void *)&zero;
    int best = -1;
    float bestMs = 0;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < 8; i++) {
        int64_t *t = tried + 5 * i;
        t[0] = -1;
        t[1] = t[2] = t[3] = t[4] = 0;
        if (i >= p->found) continue;
        int32_t v[4] = {0, 0, 0, 0};
        size_t got = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&p->offered[i].algo, CUBLASLT_ALGO_CONFIG_ID, &v[0], sizeof v[0], &got);
        cublasLtMatmulAlgoConfigGetAttribute(&p->offered[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &v[1], sizeof v[1], &got);
        cublasLtMatmulAlgoConfigGetAttribute(&p->offered[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &v[2], sizeof v[2], &got);
        cublasLtMatmulAlgoConfigGetAttribute(&p->offered[i].algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &v[3], sizeof v[3], &got);
        for (int j = 0; j < 4; j++) t[j + 1] = v[j];
    }
    for (int i = 0; i < p->found && ce == cudaSuccess; i++) {
        if (p->offered[i].workspaceSize > workBytes) continue;
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        for (int pass = 0; pass < 2 && st == CUBLAS_STATUS_SUCCESS; pass++) {
            if (pass == 1) cudaEventRecord(e0, s);
            for (int r = 0; r < reps && st == CUBLAS_STATUS_SUCCESS; r++)
                for (int32_t j = 0; j < outer && st == CUBLAS_STATUS_SUCCESS; j++) {
                    void *cj = (void *)(uintptr_t)(c + (uint64_t)(j * oc));
                    st = cublasLtMatmul((cublasLtHandle_t)(uintptr_t)lt, p->op, pa, (const void *)(uintptr_t)(a + (uint64_t)(j * oa)),
                                        p->a, (const void *)(uintptr_t)(b + (uint64_t)(j * ob)), p->b, pz, cj, p->c, cj, p->c,
                                        &p->offered[i].algo, (void *)(uintptr_t)work, (size_t)workBytes, s);
                }
        }
        cudaEventRecord(e1, s);
        ce = cudaEventSynchronize(e1);
        if (st != CUBLAS_STATUS_SUCCESS) {
            if (status == CUBLAS_STATUS_SUCCESS) status = st;
            continue;
        }
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        tried[5 * i] = (int64_t)((double)ms * 1e6 / reps);
        if (best < 0 || ms < bestMs) {
            best = i;
            bestMs = ms;
        }
    }
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (ce != cudaSuccess) return -(int32_t)ce;
    if (best < 0) return status != CUBLAS_STATUS_SUCCESS ? (int32_t)status : CUBLAS_STATUS_NOT_SUPPORTED;
    choose(p, best, info, (int64_t)((double)bestMs * 1e6 / reps));
    return CUBLAS_STATUS_SUCCESS;
}

/* a widening: a matrix of rows x cols (column-major, ld) of type from into a contiguous one (ld rows) of type to, by
 * cublasLtMatrixTransform - BF16 and F16 to F32 exactly */
typedef struct {
    cublasLtMatrixTransformDesc_t t;
    cublasLtMatrixLayout_t from, to;
} oann_widen;

static void widen_free(oann_widen *w) {
    if (!w) return;
    if (w->to) cublasLtMatrixLayoutDestroy(w->to);
    if (w->from) cublasLtMatrixLayoutDestroy(w->from);
    if (w->t) cublasLtMatrixTransformDescDestroy(w->t);
    free(w);
}

void oann_widen_free(uint64_t w) { widen_free((oann_widen *)(uintptr_t)w); }

int32_t oann_widen_plan(uint64_t *out, int32_t from, int32_t rows, int32_t cols, int64_t ld, int32_t to) {
    *out = 0;
    oann_widen *w = (oann_widen *)calloc(1, sizeof(oann_widen));
    if (!w) return CUBLAS_STATUS_ALLOC_FAILED;
    cublasStatus_t s = cublasLtMatrixTransformDescCreate(&w->t, CUDA_R_32F);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutCreate(&w->from, (cudaDataType)from, (uint64_t)rows, (uint64_t)cols, ld);
    if (s == CUBLAS_STATUS_SUCCESS) s = cublasLtMatrixLayoutCreate(&w->to, (cudaDataType)to, (uint64_t)rows, (uint64_t)cols, rows);
    if (s != CUBLAS_STATUS_SUCCESS) {
        widen_free(w);
        return s;
    }
    *out = (uint64_t)(uintptr_t)w;
    return CUBLAS_STATUS_SUCCESS;
}

void oann_widen_run(uint64_t st, uint64_t lt, uint64_t plan, uint64_t src, uint64_t dst, uint64_t stream) {
    oann_widen *w = (oann_widen *)(uintptr_t)plan;
    float one = 1, zero = 0;
    note(st, OANN_CUBLAS,
         cublasLtMatrixTransform((cublasLtHandle_t)(uintptr_t)lt, w->t, &one, (const void *)(uintptr_t)src, w->from, &zero,
                                 NULL, NULL, (void *)(uintptr_t)dst, w->to, (cudaStream_t)(uintptr_t)stream),
         "cublasLtMatrixTransform (widening)");
}

/* C = diag(x) C, column-major rows x cols (ld): row i scaled by x[i] - an INT8 product's per-output scale - in place */
void oann_scale_rows(uint64_t st, uint64_t h, int32_t rows, int32_t cols, uint64_t c, int64_t ldc, uint64_t x) {
    float *pc = (float *)(uintptr_t)c;
    note(st, OANN_CUBLAS,
         cublasSdgmm((cublasHandle_t)(uintptr_t)h, CUBLAS_SIDE_LEFT, rows, cols, pc, (int)ldc, (const float *)(uintptr_t)x, 1, pc, (int)ldc),
         "cublasSdgmm (INT8 scale)");
}

/* stream capture, for the tests: what is enqueued between begin and end is recorded into a CUDA graph rather than run
 * (thread-local mode: an allocation or another call unsafe in a capture fails it); end replays the graph `launches`
 * times on the stream and frees it, giving the graph's node count in nodes[0]. Each returns its cudaError_t */
int32_t oann_capture_begin(uint64_t stream) {
    return cudaStreamBeginCapture((cudaStream_t)(uintptr_t)stream, cudaStreamCaptureModeThreadLocal);
}

int32_t oann_capture_end(uint64_t stream, int32_t launches, int32_t *nodes) {
    cudaStream_t s = (cudaStream_t)(uintptr_t)stream;
    cudaGraph_t g = 0;
    cudaGraphExec_t x = 0;
    cudaError_t e = cudaStreamEndCapture(s, &g);
    if (e != cudaSuccess) return e;
    size_t count = 0;
    e = cudaGraphGetNodes(g, NULL, &count);
    nodes[0] = (int32_t)count;
    if (e == cudaSuccess) e = cudaGraphInstantiate(&x, g, 0);
    for (int32_t i = 0; i < launches && e == cudaSuccess; i++) e = cudaGraphLaunch(x, s);
    if (e == cudaSuccess) e = cudaStreamSynchronize(s);
    if (x) cudaGraphExecDestroy(x);
    cudaGraphDestroy(g);
    return e;
}

/* GP7: a capture kept and replayed. end_keep ends the capture begun by oann_capture_begin and instantiates the graph into
 * *exec (nodes[0] its node count); launch enqueues one replay on the stream (a failure into the status record); free
 * destroys it. What was enqueued during the capture is NOT run by the capture: the caller launches the graph */
int32_t oann_graph_end_keep(uint64_t stream, uint64_t *exec, int32_t *nodes) {
    cudaGraph_t g = 0;
    cudaGraphExec_t x = 0;
    *exec = 0;
    cudaError_t e = cudaStreamEndCapture((cudaStream_t)(uintptr_t)stream, &g);
    if (e != cudaSuccess) return e;
    size_t count = 0;
    e = cudaGraphGetNodes(g, NULL, &count);
    nodes[0] = (int32_t)count;
    if (e == cudaSuccess) e = cudaGraphInstantiate(&x, g, 0);
    cudaGraphDestroy(g);
    if (e == cudaSuccess) *exec = (uint64_t)(uintptr_t)x;
    return e;
}

void oann_graph_launch(uint64_t st, uint64_t exec, uint64_t stream) {
    note(st, OANN_CUDA, cudaGraphLaunch((cudaGraphExec_t)(uintptr_t)exec, (cudaStream_t)(uintptr_t)stream), "cudaGraphLaunch");
}

void oann_graph_free(uint64_t exec) { cudaGraphExecDestroy((cudaGraphExec_t)(uintptr_t)exec); }

/* ---- GP4 (dyn/mirror.olang, the settle on the GPU): pinned host staging, copies through it, events waited on ----
 *
 * A plan stages a batch's inputs (drives, labels, targets, the phases' steps) into pinned memory it allocated at Plan
 * and uploads them in one asynchronous copy; the settle's residual words come down into pinned memory too, after an
 * event, so the host reads them while the device runs the sweeps queued after them. Pinned memory is addressed by U64
 * like device memory; the host reads and writes it only through these functions. */

int32_t oann_pinned_alloc(uint64_t *out, uint64_t bytes) {
    void *p = 0;
    cudaError_t e = cudaHostAlloc(&p, (size_t)(bytes > 0 ? bytes : 1), cudaHostAllocDefault);
    *out = (uint64_t)(uintptr_t)p;
    return e;
}

int32_t oann_pinned_free(uint64_t p) { return cudaFreeHost((void *)(uintptr_t)p); }

/* bytes of host from host + from (bytes) into pinned memory at pinned + at */
void oann_stage(uint64_t pinned, uint64_t at, const void *host, uint64_t from, uint64_t bytes) {
    memcpy((char *)(uintptr_t)pinned + at, (const char *)host + from, (size_t)bytes);
}

/* the 64-bit word i of pinned memory */
uint64_t oann_pinned_word(uint64_t pinned, uint64_t i) { return ((const volatile uint64_t *)(uintptr_t)pinned)[i]; }

/* bytes from pinned memory at src to the device address dst on the copy stream, ordered before what is enqueued next
 * on the compute stream by the event (as oann_upload, but asynchronous to the host: src must not be written until the
 * compute stream has passed it) */
void oann_upload_pinned(uint64_t st, uint64_t dst, uint64_t src, uint64_t bytes, uint64_t copy, uint64_t event,
                        uint64_t compute) {
    cudaStream_t cs = (cudaStream_t)(uintptr_t)copy;
    note(st, OANN_CUDA, cudaMemcpyAsync((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)bytes,
                                        cudaMemcpyHostToDevice, cs), "cudaMemcpyAsync (staged upload)");
    note(st, OANN_CUDA, cudaEventRecord((cudaEvent_t)(uintptr_t)event, cs), "cudaEventRecord (staged upload)");
    note(st, OANN_CUDA, cudaStreamWaitEvent((cudaStream_t)(uintptr_t)compute, (cudaEvent_t)(uintptr_t)event, 0),
         "cudaStreamWaitEvent (staged upload)");
}

/* bytes from the device address src to pinned memory at dst, on the compute stream, then the event recorded there */
void oann_download_pinned(uint64_t st, uint64_t dst, uint64_t src, uint64_t bytes, uint64_t compute, uint64_t event) {
    cudaStream_t cs = (cudaStream_t)(uintptr_t)compute;
    note(st, OANN_CUDA, cudaMemcpyAsync((void *)(uintptr_t)dst, (const void *)(uintptr_t)src, (size_t)bytes,
                                        cudaMemcpyDeviceToHost, cs), "cudaMemcpyAsync (pinned download)");
    note(st, OANN_CUDA, cudaEventRecord((cudaEvent_t)(uintptr_t)event, cs), "cudaEventRecord (pinned download)");
}

/* waits for the event; its status (an asynchronous fault shows here) */
int32_t oann_event_sync(uint64_t ev) { return cudaEventSynchronize((cudaEvent_t)(uintptr_t)ev); }
