// elem.cu - our element kernels on the GPU, and the device functions the other kernels share (docs/redesign.md 1.10,
// "Our own kernels"). CUDA C text, compiled by NVRTC when a plan is made (kernels/gpu/kern.olang), never by a build.
//
// kern.olang puts before this text the #defines it reads:
//   OANN_E32_LO, _HI, _LOG2E, _SHIFT, _LN2HI, _LN2LO, OANN_E32_C0 .. C6 - kernels/elem's fastExp32's range, its
//     reduction's constants and its Taylor coefficients (elem.ExpTaylor rounded to F32), each as its bits;
//   OANN_E64_... and OANN_E64_C0 .. C12 - the same for fastExp64.
// So FastExp and FastTanh here are kernels/elem's, operation for operation: compiled without contraction into fused
// multiply-adds (--fmad=false, as olang compiles the CPU's) and with IEEE division, they give the CPU's bits, and in
// any case its stated error (FastExp 3.0e-7 relative, FastTanh 2.0e-7 absolute in F32).
//
// Kernels are extern "C", so their names are as written; each takes its arguments as kern.olang's Launch packs them
// (device addresses, numbers by their bits, 64-bit counts).

#define OANN_INF32 __int_as_float(0x7f800000)
#define OANN_INF64 __longlong_as_double(0x7ff0000000000000ll)

// e^x as kernels/elem's fastExp32: x = k ln 2 + r, e^r by its Taylor series to r^6, 2^k from its bits
__device__ __forceinline__ float oann_exp(float x) {
    float t = x > OANN_E32_LO ? x : OANN_E32_LO;
    t = t < OANN_E32_HI ? t : OANN_E32_HI;
    float kb = t * OANN_E32_LOG2E + OANN_E32_SHIFT;
    float k = kb - OANN_E32_SHIFT;
    unsigned ki = __float_as_uint(kb) - 0x4B400000u;
    float r = t - k * OANN_E32_LN2HI - k * OANN_E32_LN2LO;
    float p = OANN_E32_C6;
    p = p * r + OANN_E32_C5;
    p = p * r + OANN_E32_C4;
    p = p * r + OANN_E32_C3;
    p = p * r + OANN_E32_C2;
    p = p * r + OANN_E32_C1;
    p = p * r + OANN_E32_C0;
    bool up = k > 0.0f;
    float scale = __uint_as_float((ki - (up ? 1u : 0u) + 127u) << 23);
    float e = p * scale * (up ? 2.0f : 1.0f);
    e = x < OANN_E32_HI ? e : OANN_INF32;
    e = x > OANN_E32_LO ? e : 0.0f;
    return x == x ? e : x;
}

// e^x as kernels/elem's fastExp64: the series to r^12
__device__ __forceinline__ double oann_exp(double x) {
    double t = x > OANN_E64_LO ? x : OANN_E64_LO;
    t = t < OANN_E64_HI ? t : OANN_E64_HI;
    double kb = t * OANN_E64_LOG2E + OANN_E64_SHIFT;
    double k = kb - OANN_E64_SHIFT;
    unsigned long long ki = (unsigned long long)__double_as_longlong(kb) - 0x4338000000000000ull;
    double r = t - k * OANN_E64_LN2HI - k * OANN_E64_LN2LO;
    double p = OANN_E64_C12;
    p = p * r + OANN_E64_C11;
    p = p * r + OANN_E64_C10;
    p = p * r + OANN_E64_C9;
    p = p * r + OANN_E64_C8;
    p = p * r + OANN_E64_C7;
    p = p * r + OANN_E64_C6;
    p = p * r + OANN_E64_C5;
    p = p * r + OANN_E64_C4;
    p = p * r + OANN_E64_C3;
    p = p * r + OANN_E64_C2;
    p = p * r + OANN_E64_C1;
    p = p * r + OANN_E64_C0;
    bool up = k > 0.0;
    double scale = __longlong_as_double((long long)((ki - (up ? 1ull : 0ull) + 1023ull) << 52));
    double e = p * scale * (up ? 2.0 : 1.0);
    e = x < OANN_E64_HI ? e : OANN_INF64;
    e = x > OANN_E64_LO ? e : 0.0;
    return x == x ? e : x;
}

// tanh as kernels/elem's FastTanh: 1 - 2 / (e^(2|x|) + 1) with x's sign, |x| held at 10 (F32) or 20 (F64)
__device__ __forceinline__ float oann_tanh(float x) {
    float a = fabsf(x);
    a = a > 10.0f ? 10.0f : a;
    float v = 1.0f - 2.0f / (oann_exp(a + a) + 1.0f);
    return copysignf(v, x);
}

__device__ __forceinline__ double oann_tanh(double x) {
    double a = fabs(x);
    a = a > 20.0 ? 20.0 : a;
    double v = 1.0 - 2.0 / (oann_exp(a + a) + 1.0);
    return copysign(v, x);
}

// max(x, 0) as olang's Max (IEEE 754's maximum): a NaN stays one, -0 gives +0 (fmaxf would give 0 for a NaN)
template <typename T> __device__ __forceinline__ T oann_relu(T x) { return x != x ? x : (x > T(0) ? x : T(0)); }

// clip(x, 0, 1) as dyn/neuron's clip: -0 stays -0, a NaN stays one
template <typename T> __device__ __forceinline__ T oann_clip(T x) { return x < T(0) ? T(0) : (x > T(1) ? T(1) : x); }

// the steady firing rate of a Lif neuron, dyn/neuron's LifRateOf, in F64; tau = -1 / ln(1 - leak) is the host's
__device__ __forceinline__ double oann_lif_rate(double x, double threshold, double tau) {
    if (x <= threshold) return 0.0;
    double period = tau * log(x / (x - threshold));
    return period <= 1.0 ? 1.0 : 1.0 / period;
}

// y[i] += alpha x[i] for i < n: kernels/elem's Axpy, which a launch's argument packing is tested on (an address, a
// 32- or 64-bit float by its bits, an address, a 64-bit count)
extern "C" __global__ void oann_axpy_f32(float *y, float alpha, const float *x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] += alpha * x[i];
}

extern "C" __global__ void oann_axpy_f64(double *y, double alpha, const double *x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] += alpha * x[i];
}

// y[i] = tanh(x[i]) by oann_tanh, for checking FastTanh on the GPU against kernels/elem's at many points
extern "C" __global__ void oann_tanh_f32(float *y, const float *x, long long n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x)
        y[i] = oann_tanh(x[i]);
}

// AdamW's step over n parameters, nn/optim's adamRun operation for operation (so bit for bit, with IEEE square roots
// and division and no contraction): m = b1 m + c1 g, v = b2 v + c2 g g, p = keep p - step m / (sqrt(v) root + eps)
#define OANN_ADAMW(name, T, SQRT)                                                                                     \
    extern "C" __global__ void name(T *p, const T *g, T *m, T *v, long long n, T b1, T b2, T c1, T c2, T keep,     \
                                    T step, T root, T eps) {                                                        \
        for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < n;                                  \
             k += (long long)gridDim.x * blockDim.x) {                                                               \
            T gk = g[k];                                                                                             \
            T mk = b1 * m[k] + c1 * gk;                                                                              \
            T vk = b2 * v[k] + c2 * gk * gk;                                                                         \
            m[k] = mk;                                                                                               \
            v[k] = vk;                                                                                               \
            p[k] = keep * p[k] - step * mk / (SQRT(vk) * root + eps);                                                \
        }                                                                                                            \
    }
OANN_ADAMW(oann_adamw_f32, float, sqrtf)
OANN_ADAMW(oann_adamw_f64, double, sqrt)
