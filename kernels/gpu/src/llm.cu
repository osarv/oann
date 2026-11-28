// llm.cu - the kernels of llm/gpu.olang, a language model's decoding step on the GPU: everything between the products
// (cuBLASLt's). CUDA C text compiled by NVRTC after elem.cu (oann_exp is kernels/elem's FastExp).
//
// Storage. The residual stream and the products' results are F32; what a product reads (a normed row, the attention's
// result, the perceptron's gated hidden row), the weights and the cached keys and values are BF16, rounded to nearest,
// even, as the CPU's path rounds them.
//
// The cache. Keys and values of layer l, page p, key-value head g: 64 * dh elements from ((l * pages + p) * kv + g) *
// 64 * dh. Position j's value is the dh elements at j * dh; its key's dimension i is at i * 64 + j (dimension by
// dimension, so the threads of the attention's scores, one a position, read neighbours). A sequence's pages are its slot's row of the table (maxPages a row):
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
        at = ((layerPages + page) * kv + g) * LLM_PAGE * dh;
    }
    long long n = pos[r] % LLM_PAGE;
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
        else if (isV) V[at + n * dh + i] = llm_round(v);
        else K[at + i * LLM_PAGE + n] = llm_round(v);
    }
}

// attention, split by pages (flash decoding). llm_attend: a block for each (row r, key-value head g, page c of r's
// sequence), item = (r * kv + g) * chunks + c, for the heads / kv (at most 8) query heads that read g, h = g * (heads /
// kv) + t: the page's scores q . k * scale, their maximum m, sum s of exp(score - m) and the values weighted by them,
// a, into part[(r * heads + h) * chunks + c] as (m, s, a[0 .. dh)). The page's keys and values are read once for the
// group's heads; nothing is written for a page past last[r]. llm_combine: a block for each (row, head) joins its
// pages' parts into out[r, h, :] (BF16)
#define LLM_GROUP 8
extern "C" __global__ void llm_attend(float *part, const unsigned short *q, const unsigned short *K, const unsigned short *V,
                                      const int *last, const int *slot, const int *table, long long heads, long long kv,
                                      long long dh, long long layerPages, long long maxPages, long long chunks, float scale) {
    __shared__ float qs[LLM_GROUP * 256];
    __shared__ float w[LLM_GROUP * LLM_PAGE];
    __shared__ float stat[2 * LLM_GROUP];
    int nq = (int)(heads / kv), D = (int)dh, tid = threadIdx.x, bd = blockDim.x;
    long long c = blockIdx.x % chunks, rg = blockIdx.x / chunks, r = rg / kv, g = rg % kv;
    int np = last[r] + 1;
    int first = (int)c * LLM_PAGE;
    if (first >= np) return;
    int cnt = np - first < LLM_PAGE ? np - first : LLM_PAGE;
    long long base = ((layerPages + table[(long long)slot[r] * maxPages + c]) * kv + g) * LLM_PAGE * dh;
    const unsigned short *qv = q + (r * heads + g * nq) * dh;
    for (int i = tid; i < nq * D; i += bd) qs[i] = llm_bf(qv[i]);
    __syncthreads();
    // the scores: item (t, j), keys stored dimension by dimension so neighbouring j read neighbours
    for (int it = tid; it < nq * LLM_PAGE; it += bd) {
        int t = it / LLM_PAGE, j = it % LLM_PAGE;
        float a = -OANN_INF32;
        if (j < cnt) {
            a = 0.0f;
            const unsigned short *kp = K + base + j;
            const float *qt = qs + t * D;
            for (int i = 0; i < D; i++) a += qt[i] * llm_bf(kp[i * LLM_PAGE]);
            a *= scale;
        }
        w[it] = a;
    }
    __syncthreads();
    // each head's maximum and sum over the page (a warp a head, nq <= 8 warps of the block's)
    int warp = tid / 32, lane = tid % 32;
    for (int t = warp; t < nq; t += bd / 32) {
        float m = -OANN_INF32;
        for (int j = lane; j < cnt; j += 32) m = fmaxf(m, w[t * LLM_PAGE + j]);
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float sum = 0.0f;
        for (int j = lane; j < LLM_PAGE; j += 32) {
            float e = j < cnt ? oann_exp(w[t * LLM_PAGE + j] - m) : 0.0f;
            w[t * LLM_PAGE + j] = e;
            sum += e;
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) {
            stat[2 * t] = m;
            stat[2 * t + 1] = sum;
        }
    }
    __syncthreads();
    // the weighted values: a thread a dimension, values stored position by position
    for (int i = tid; i < D; i += bd) {
        float acc[LLM_GROUP];
        for (int t = 0; t < nq; t++) acc[t] = 0.0f;
        const unsigned short *vp = V + base + i;
        for (int j = 0; j < cnt; j++) {
            float vx = llm_bf(vp[(long long)j * D]);
            for (int t = 0; t < nq; t++) acc[t] += w[t * LLM_PAGE + j] * vx;
        }
        for (int t = 0; t < nq; t++) part[((r * heads + g * nq + t) * chunks + c) * (dh + 2) + 2 + i] = acc[t];
    }
    if (tid < nq) {
        float *pp = part + ((r * heads + g * nq + tid) * chunks + c) * (dh + 2);
        pp[0] = stat[2 * tid];
        pp[1] = stat[2 * tid + 1];
    }
}

extern "C" __global__ void llm_combine(unsigned short *out, const float *part, const int *last, long long heads, long long dh, long long chunks) {
    long long item = blockIdx.x, r = item / heads;
    int nc = (last[r] + LLM_PAGE) / LLM_PAGE;
    const float *pp = part + item * chunks * (dh + 2);
    float m = -OANN_INF32;
    for (int c = 0; c < nc; c++) m = fmaxf(m, pp[c * (dh + 2)]);
    for (long long i = threadIdx.x; i < dh; i += blockDim.x) {
        float s = 0.0f, a = 0.0f;
        for (int c = 0; c < nc; c++) {
            const float *q = pp + c * (dh + 2);
            float k = oann_exp(q[0] - m);
            s += q[1] * k;
            a += q[2 + i] * k;
        }
        out[item * dh + i] = llm_round(a / s);
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
