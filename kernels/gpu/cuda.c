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
