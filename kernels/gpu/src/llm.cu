// llm.cu - the kernels of llm/gpu.olang, a language model's decoding step on the GPU: everything between the products
// (cuBLASLt's). CUDA C text compiled by NVRTC after elem.cu (oann_exp is kernels/elem's FastExp).
//
// Storage. The residual stream and the products' results are F32; what a product reads (a normed row, the attention's
// result, the perceptron's gated hidden row), the weights and the cached keys and values are BF16, rounded to nearest,
// even, as the CPU's path rounds them.
//
// The cache. Keys and values of layer l, page p, key-value head g, position j of the page, HeadDim each, at
// (((l * pages + p) * kv + g) * 64 + j) * dh. A sequence's pages are its slot's row of the table (maxPages a row):
// position n is in page table[slot * maxPages + n / 64], at n % 64.

#define LLM_LOOP(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)
#define LLM_PAGE 64

__device__ __forceinline__ float llm_bf(unsigned short b) { return __uint_as_float(((unsigned)b) << 16); }

__device__ __forceinline__ unsigned short llm_round(float x) {
    unsigned u = __float_as_uint(x);
    return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// x[r, :] = table[tok[r], :] * scale
extern "C" __global__ void llm_embed(float *x, const int *tok, const unsigned short *table, long long n, long long d, float scale) {
    LLM_LOOP(i, n * d) {
        long long r = i / d;
        x[i] = llm_bf(table[(long long)tok[r] * d + i % d]) * scale;
    }
}

// y[j, :] (BF16) = x[row, :] * gain / rms(x[row, :]), row = rows[j] where rows is given, else j: a block a row, the
// sum of squares in F64
extern "C" __global__ void llm_norm(unsigned short *y, const float *x, const float *gain, const int *rows, long long d, double eps) {
    __shared__ double part[256];
    long long j = blockIdx.x;
    const float *src = x + (rows ? (long long)rows[j] : j) * d;
    double s = 0.0;
    for (long long i = threadIdx.x; i < d; i += blockDim.x) s += (double)src[i] * (double)src[i];
    part[threadIdx.x] = s;
    __syncthreads();
    for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    float inv = (float)(1.0 / sqrt(part[0] / (double)d + eps));
    for (long long i = threadIdx.x; i < d; i += blockDim.x) y[j * d + i] = llm_round(src[i] * inv * gain[i]);
}

// head hh of row r (a block each, item = r * (heads + 2 kv) + hh: the queries' heads, then the keys', then the
// values'): a query or key normed over its head (QK-norm: times gain / rms, qn and kn the gains; qn 0: none), then
// rotated by the row's angles (pairs i and i + half; rot 0: none). The queries go into q (BF16, heads * dh a row), the
// keys and the values into the row's place in the cache
extern "C" __global__ void llm_heads(unsigned short *q, unsigned short *K, unsigned short *V, const float *qkv, const float *cs,
                                     const float *sn, const int *pos, const int *slot, const int *table, const float *qn,
                                     const float *kn, long long heads, long long kv, long long dh, long long half, long long layerPages,
                                     long long maxPages, long long rot, double eps) {
    __shared__ double part[256];
    long long hc = heads + 2 * kv, r = blockIdx.x / hc, hh = blockIdx.x % hc;
    const float *x = qkv + (r * hc + hh) * dh;
    bool qk = hh < heads + kv, normed = qn && qk;
    const float *gain = hh < heads ? qn : kn;
    float inv = 1.0f;
    if (normed) {
        double s = 0.0;
        for (long long i = threadIdx.x; i < dh; i += blockDim.x) s += (double)x[i] * (double)x[i];
        part[threadIdx.x] = s;
        __syncthreads();
        for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
            if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
            __syncthreads();
        }
        inv = (float)(1.0 / sqrt(part[0] / (double)dh + eps));
    }
    long long g = hh - heads, isV = g >= kv;
    if (isV) g -= kv;
    long long at = 0;
    if (!(hh < heads)) {
        long long page = table[(long long)slot[r] * maxPages + pos[r] / LLM_PAGE];
        at = (((layerPages + page) * kv + g) * LLM_PAGE + pos[r] % LLM_PAGE) * dh;
    }
    for (long long i = threadIdx.x; i < dh; i += blockDim.x) {
        float v = normed ? x[i] * inv * gain[i] : x[i];
        if (rot && qk && i < 2 * half) {
            long long j = i < half ? i : i - half;
            float lo = x[j], hi = x[half + j];
            if (normed) {
                lo = lo * inv * gain[j];
                hi = hi * inv * gain[half + j];
            }
            float c = cs[r * half + j], sv = sn[r * half + j];
            v = i < half ? lo * c - hi * sv : hi * c + lo * sv;
        }
        if (hh < heads) q[(r * heads + hh) * dh + i] = llm_round(v);
        else (isV ? V : K)[at + i] = llm_round(v);
    }
}

// attention of row r's head h (a block each, block item = r * heads + h) over positions 0 .. last[r] of its sequence:
// the scores q . k * scale into scores[item, :], their softmax, and out[r, h, :] (BF16) = the values' mean by it.
// Head h reads key-value head h / (heads / kv)
extern "C" __global__ void llm_attend(unsigned short *out, const unsigned short *q, const unsigned short *K, const unsigned short *V,
                                      float *scores, const int *last, const int *slot, const int *table, long long heads, long long kv,
                                      long long dh, long long layerPages, long long maxPages, long long cap, float scale) {
    __shared__ float part[256];
    long long item = blockIdx.x, r = item / heads, h = item % heads, g = h / (heads / kv);
    long long np = (long long)last[r] + 1;
    const int *pages = table + (long long)slot[r] * maxPages;
    const unsigned short *qv = q + (r * heads + h) * dh;
    float *sc = scores + item * cap;
    float top = -OANN_INF32;
    for (long long p = threadIdx.x; p < np; p += blockDim.x) {
        const unsigned short *kp = K + (((layerPages + pages[p / LLM_PAGE]) * kv + g) * LLM_PAGE + p % LLM_PAGE) * dh;
        float s = 0.0f;
        for (long long i = 0; i < dh; i++) s += llm_bf(qv[i]) * llm_bf(kp[i]);
        s *= scale;
        sc[p] = s;
        top = s > top ? s : top;
    }
    part[threadIdx.x] = top;
    __syncthreads();
    for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] = part[threadIdx.x] > part[threadIdx.x + w] ? part[threadIdx.x] : part[threadIdx.x + w];
        __syncthreads();
    }
    top = part[0];
    __syncthreads();
    float sum = 0.0f;
    for (long long p = threadIdx.x; p < np; p += blockDim.x) {
        float e = oann_exp(sc[p] - top);
        sc[p] = e;
        sum += e;
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned w = blockDim.x / 2; w > 0; w >>= 1) {
        if (threadIdx.x < w) part[threadIdx.x] += part[threadIdx.x + w];
        __syncthreads();
    }
    float inv = 1.0f / part[0];
    for (long long i = threadIdx.x; i < dh; i += blockDim.x) {
        float a = 0.0f;
        for (long long pg = 0; pg * LLM_PAGE < np; pg++) {
            const unsigned short *vp = V + ((layerPages + pages[pg]) * kv + g) * LLM_PAGE * dh + i;
            long long cnt = np - pg * LLM_PAGE < LLM_PAGE ? np - pg * LLM_PAGE : LLM_PAGE;
            const float *w = sc + pg * LLM_PAGE;
            for (long long j = 0; j < cnt; j++) a += w[j] * llm_bf(vp[j * dh]);
        }
        out[(r * heads + h) * dh + i] = llm_round(a * inv);
    }
}

// x += y * rs
extern "C" __global__ void llm_add(float *x, const float *y, long long n, float rs) { LLM_LOOP(i, n) x[i] += y[i] * rs; }

// h[r, i] (BF16) = silu(y[r, i]) * y[r, f + i], y's rows 2 f wide (the gate, then the up projection)
extern "C" __global__ void llm_swiglu(unsigned short *h, const float *y, long long n, long long f) {
    LLM_LOOP(e, n * f) {
        long long r = e / f, i = e % f;
        float gt = y[r * 2 * f + i];
        h[e] = llm_round(gt / (1.0f + oann_exp(-gt)) * y[r * 2 * f + f + i]);
    }
}

// y[r, c] += b[c]
extern "C" __global__ void llm_bias(float *y, const float *b, long long n, long long cols) { LLM_LOOP(i, n * cols) y[i] += b[i % cols]; }

// y *= k
extern "C" __global__ void llm_scale(float *y, long long n, float k) { LLM_LOOP(i, n) y[i] *= k; }
