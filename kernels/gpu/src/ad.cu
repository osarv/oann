// ad.cu - the element kernels of ad/graph on the GPU (docs/redesign.md 1.10, GP6): the activations and their slopes,
// softmax and log-sum-exp, the norms, the losses, lookups, sums and AdamW, each the CPU's (kernels/elem, ad/ops) over
// the graph's arena. CUDA C text compiled by NVRTC when a graph is planned on the GPU (ad/mirror.olang), after elem.cu's
// text (kern.olang's prelude: oann_exp, oann_tanh, oann_relu) and these #defines:
//   real      float or double, the graph's element type (typedef)
//
// Every operand is a dense row-major run (the graph's invariant for what a kernel reads: rows of `cols` elements one
// after another), so a kernel takes an address and counts, never a stride. Counts and flags are 64-bit; reals are
// `real`. Compiled without contraction (--fmad=false) with IEEE division and square roots, so an element-wise kernel is
// the CPU's operation for operation - the exponential, the tanh and the sigmoid are elem.cu's bit for bit - and the
// reductions differ from the CPU's in the order they sum (a block's lanes, not eight lanes), within the usual
// tolerance. Reductions are deterministic: every block reduces in one fixed order, and no kernel here uses atomics.
//
// The gradient kernels take `add`: 1 where the gradient is added to what is there (the second contribution to it),
// 0 where it is written. Row kernels run one block a row (block a power of two, 32 to 256); element kernels stride a
// grid over their elements.

#define OANN_LOOP(i, n) for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

// ---- block reductions: the sum or maximum of every thread's v, given to every thread; blockDim a multiple of 32 ----

template <typename V> __device__ __forceinline__ V oann_bsum(V v) {
    __shared__ V part[32];
    __shared__ V total;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        V t = lane < (int)(blockDim.x >> 5) ? part[lane] : V(0);
        for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
        if (lane == 0) total = t;
    }
    __syncthreads();
    return total;
}

template <typename V> __device__ __forceinline__ V oann_bmax(V v) {
    __shared__ V part[32];
    __shared__ V top;
    for (int o = 16; o > 0; o >>= 1) {
        V w = __shfl_down_sync(0xffffffffu, v, o);
        v = w > v ? w : v;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        V t = lane < (int)(blockDim.x >> 5) ? part[lane] : part[0];
        for (int o = 16; o > 0; o >>= 1) {
            V w = __shfl_down_sync(0xffffffffu, t, o);
            t = w > t ? w : t;
        }
        if (lane == 0) top = t;
    }
    __syncthreads();
    return top;
}

// ---- activations (kernels/elem's, by mat.Act's order: Relu Gelu GeluTanh Silu Sigmoid Tanh Softplus) ----

__device__ __forceinline__ real oann_sigmoid(real x) { return (real)1.0 / ((real)1.0 + oann_exp(-x)); }

__device__ __forceinline__ real oann_gelu_arg(real x) { return (real)0.7978845608028654 * (x + (real)0.044715 * x * x * x); }

__device__ __forceinline__ real oann_act(real x, int a) {
    switch (a) {
    case 0: return x < (real)0 ? (real)0 : x;
    case 1: return (real)0.5 * x * erfc(-x * (real)0.7071067811865476);
    case 2: return x / ((real)1.0 + oann_exp((real)-2.0 * oann_gelu_arg(x)));
    case 3: return x * oann_sigmoid(x);
    case 4: return oann_sigmoid(x);
    case 5: return oann_tanh(x);
    default: return (x > (real)0 ? x : (real)0) + log1p(oann_exp(-fabs(x)));
    }
}

// act'(x), from the activation's input
__device__ __forceinline__ real oann_slope(real x, int a) {
    switch (a) {
    case 0: return x > (real)0 ? (real)1.0 : (real)0.0;
    case 1: {
        real cdf = (real)0.5 * erfc(-x * (real)0.7071067811865476);
        real pdf = (real)0.3989422804014327 * exp((real)-0.5 * x * x);
        return cdf + x * pdf;
    }
    case 2: {
        real s = (real)1.0 / ((real)1.0 + oann_exp((real)-2.0 * oann_gelu_arg(x)));
        return s + (real)2.0 * x * s * ((real)1.0 - s) * (real)0.7978845608028654 * ((real)1.0 + (real)0.134145 * x * x);
    }
    case 3: {
        real s = oann_sigmoid(x);
        return s * ((real)1.0 + x * ((real)1.0 - s));
    }
    case 4: {
        real s = oann_sigmoid(x);
        return s * ((real)1.0 - s);
    }
    case 5: {
        real t = oann_tanh(x);
        return (real)1.0 - t * t;
    }
    default: return oann_sigmoid(x);
    }
}

// act'(x) from y = act(x), for Relu, Sigmoid, Tanh (kernels/elem's FromOutput)
__device__ __forceinline__ real oann_slope_out(real y, int a) {
    switch (a) {
    case 0: return y > (real)0 ? (real)1.0 : (real)0.0;
    case 4: return y * ((real)1.0 - y);
    default: return (real)1.0 - y * y;
    }
}

extern "C" __global__ void oann_ad_act(real *y, const real *x, long long n, long long a) {
    OANN_LOOP(i, n) y[i] = oann_act(x[i], (int)a);
}

// dx (+)= dy act'(x), the slope read from x (fromOut 0) or from y = act(x) (fromOut 1, Relu Sigmoid Tanh)
extern "C" __global__ void oann_ad_act_grad(real *dx, const real *dy, const real *src, long long n, long long a, long long fromOut, long long add) {
    OANN_LOOP(i, n) {
        real s = fromOut ? oann_slope_out(src[i], (int)a) : oann_slope(src[i], (int)a);
        real v;
        if (fromOut && a == 0) v = src[i] > (real)0 ? dy[i] : (real)0;
        else v = dy[i] * s;
        dx[i] = add ? dx[i] + v : v;
    }
}

extern "C" __global__ void oann_ad_leaky(real *y, const real *x, long long n, real slope) {
    OANN_LOOP(i, n) { real v = x[i]; y[i] = v > (real)0 ? v : slope * v; }
}

extern "C" __global__ void oann_ad_leaky_grad(real *dx, const real *dy, const real *x, long long n, real slope, long long add) {
    OANN_LOOP(i, n) {
        real v = dy[i] * (x[i] > (real)0 ? (real)1.0 : slope);
        dx[i] = add ? dx[i] + v : v;
    }
}

// ---- element-wise arithmetic ----

extern "C" __global__ void oann_ad_fill(real *p, real v, long long n) { OANN_LOOP(i, n) p[i] = v; }

extern "C" __global__ void oann_ad_copy(real *y, const real *x, long long n) { OANN_LOOP(i, n) y[i] = x[i]; }

// dx (+)= k dy
extern "C" __global__ void oann_ad_put(real *dx, const real *dy, real k, long long n, long long add) {
    OANN_LOOP(i, n) { real v = k * dy[i]; dx[i] = add ? dx[i] + v : v; }
}

// y = a + k b
extern "C" __global__ void oann_ad_combine(real *y, const real *a, const real *b, real k, long long n) {
    OANN_LOOP(i, n) y[i] = a[i] + k * b[i];
}

extern "C" __global__ void oann_ad_mul(real *y, const real *a, const real *b, long long n) { OANN_LOOP(i, n) y[i] = a[i] * b[i]; }

// dx (+)= dy b
extern "C" __global__ void oann_ad_mul_grad(real *dx, const real *dy, const real *b, long long n, long long add) {
    OANN_LOOP(i, n) { real v = (real)1.0 * dy[i] * b[i]; dx[i] = add ? dx[i] + v : v; }
}

extern "C" __global__ void oann_ad_scale(real *y, const real *x, real k, long long n) { OANN_LOOP(i, n) y[i] = k * x[i]; }

// y[r, c] += b[c]: the bias after a product
extern "C" __global__ void oann_ad_bias(real *y, const real *b, long long n, long long cols) {
    OANN_LOOP(i, n) y[i] = y[i] + b[i % cols];
}

// out[c] = the sum of column c over the rows (+ out[c] where add): a bias's gradient. GP7: a block of 1024 threads is 32
// columns by 32 row lanes; lane l sums rows l, l + 32, ... in order, and the 32 partials are added in lane order - a fixed
// order and no atomics (the one thread a column of GP6 took 35 us for 1024 rows of 128)
extern "C" __global__ void oann_ad_colsum(real *out, const real *x, long long rows, long long cols, long long add) {
    __shared__ real part[32][33];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    long long c = (long long)blockIdx.x * 32 + tx;
    real s = (real)0;
    if (c < cols) {
        for (long long r = ty; r < rows; r += 32) s += x[r * cols + c];
    }
    part[ty][tx] = s;
    __syncthreads();
    if (ty == 0 && c < cols) {
        real t = (real)0;
        for (int i = 0; i < 32; i++) t += part[i][tx];
        out[c] = add ? t + out[c] : t;
    }
}

// ---- softmax, log-softmax and log-sum-exp over rows ----

// the band row r of a softmax reads, [*lo, *hi): the whole row (ctx 1, not causal, no window, no starts), or attention's
// weights, row r a row of a ctx x ctx matrix of sequence r / (ctx heads) at position i = r % ctx (kernels/attention's
// low and high): causal, i + 1 at most; a window W > 0, from i - W + 1; packed documents (starts: each position's document
// start, ctx a sequence), from the document's start and, not causal, to its end
__device__ __forceinline__ void oann_band(long long r, long long cols, long long ctx, long long causal, long long window, const int *starts, long long heads, long long *lo, long long *hi) {
    long long i = r % ctx;
    long long sr = (r / (ctx * heads)) * ctx + i;
    long long l = window > 0 && i - window + 1 > 0 ? i - window + 1 : 0;
    if (starts && (long long)starts[sr] > l) l = starts[sr];
    long long h = cols;
    if (causal) h = i + 1;
    else if (starts) {
        h = i + 1;
        while (h < cols && starts[sr + h - i] == starts[sr]) h++;
    }
    *lo = l;
    *hi = h;
}

// y's row r = the softmax of x's row r over its band (oann_band) and 0 outside it - attention's weights, causal, in a
// sliding window or within packed documents, or a plain softmax's row (ctx 1, no flags)
extern "C" __global__ void oann_ad_softmax(real *y, const real *x, long long cols, long long ctx, long long causal, long long window, const int *starts, long long heads) {
    long long r = blockIdx.x;
    const real *xr = x + r * cols;
    real *yr = y + r * cols;
    long long lo, last;
    oann_band(r, cols, ctx, causal, window, starts, heads, &lo, &last);
    real top = xr[lo];
    for (long long j = lo + threadIdx.x; j < last; j += blockDim.x) top = xr[j] > top ? xr[j] : top;
    top = oann_bmax(top);
    real s = (real)0;
    for (long long j = lo + threadIdx.x; j < last; j += blockDim.x) {
        real e = oann_exp(xr[j] - top);
        yr[j] = e;
        s += e;
    }
    s = oann_bsum(s);
    real scale = (real)1.0 / s;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) yr[j] = j >= lo && j < last ? yr[j] * scale : (real)0;
}

// dx (+)= scale p (dy - <dy, p>) row by row over the band (zero outside it, where written); dx may be dy
extern "C" __global__ void oann_ad_softmax_grad(real *dx, const real *dy, const real *p, long long cols, long long ctx, long long causal, long long window, const int *starts, long long heads, real scale, long long add) {
    long long r = blockIdx.x;
    const real *gr = dy + r * cols;
    const real *pr = p + r * cols;
    real *dr = dx + r * cols;
    long long lo, last;
    oann_band(r, cols, ctx, causal, window, starts, heads, &lo, &last);
    real dot = (real)0;
    for (long long j = lo + threadIdx.x; j < last; j += blockDim.x) dot += gr[j] * pr[j];
    dot = oann_bsum(dot);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        bool in = j >= lo && j < last;
        real v = in ? scale * pr[j] * (gr[j] - dot) : (real)0;
        if (in || !add) dr[j] = add ? dr[j] + v : v;
    }
}

// the log-sum-exp of each row, its largest element taken out first
__device__ __forceinline__ real oann_lse(const real *xr, long long cols) {
    real top = xr[0];
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) top = xr[j] > top ? xr[j] : top;
    top = oann_bmax(top);
    real s = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) s += oann_exp(xr[j] - top);
    s = oann_bsum(s);
    return top + log(s);
}

extern "C" __global__ void oann_ad_lse(real *out, const real *x, long long cols) {
    real v = oann_lse(x + (long long)blockIdx.x * cols, cols);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

extern "C" __global__ void oann_ad_logsoftmax(real *y, const real *x, long long cols) {
    const real *xr = x + (long long)blockIdx.x * cols;
    real *yr = y + (long long)blockIdx.x * cols;
    real l = oann_lse(xr, cols);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) yr[j] = xr[j] - l;
}

// dx (+)= dy - exp(y) sum(dy)
extern "C" __global__ void oann_ad_logsoftmax_grad(real *dx, const real *dy, const real *y, long long cols, long long add) {
    const real *gr = dy + (long long)blockIdx.x * cols;
    const real *yr = y + (long long)blockIdx.x * cols;
    real *dr = dx + (long long)blockIdx.x * cols;
    real total = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) total += gr[j];
    total = oann_bsum(total);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        real v = gr[j] - oann_exp(yr[j]) * total;
        dr[j] = add ? dr[j] + v : v;
    }
}

// ---- normalization over rows: stats = mean[rows], inv[rows] one after the other ----

extern "C" __global__ void oann_ad_norm_stats(real *mean, real *inv, const real *x, long long cols, real eps, long long rms) {
    const real *xr = x + (long long)blockIdx.x * cols;
    real n = (real)cols;
    real m = (real)0;
    if (!rms) {
        real s = (real)0;
        for (long long j = threadIdx.x; j < cols; j += blockDim.x) s += xr[j];
        m = oann_bsum(s) / n;
    }
    real q = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) q += (xr[j] - m) * (xr[j] - m);
    q = oann_bsum(q) / n;
    if (threadIdx.x == 0) {
        mean[blockIdx.x] = m;
        inv[blockIdx.x] = (real)1.0 / sqrt(q + eps);
    }
}

// y = (x - mean) inv gain + bias (bias 0: none)
extern "C" __global__ void oann_ad_norm_apply(real *y, const real *x, const real *mean, const real *inv, const real *gain, const real *bias, long long n, long long cols) {
    OANN_LOOP(i, n) {
        long long r = i / cols, c = i % cols;
        real v = (x[i] - mean[r]) * inv[r] * gain[c];
        y[i] = bias ? v + bias[c] : v;
    }
}

// dx (+)= inv (g - mean(g) - xhat mean(g xhat)), g = dy gain (mean(g) left out where rms)
extern "C" __global__ void oann_ad_norm_grad(real *dx, const real *dy, const real *x, const real *mean, const real *inv, const real *gain, long long cols, long long rms, long long add) {
    long long r = blockIdx.x;
    const real *gr = dy + r * cols;
    const real *xr = x + r * cols;
    real *dr = dx + r * cols;
    real m = mean[r], k = inv[r], n = (real)cols;
    real sg = (real)0, sgx = (real)0;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        real gc = gr[c] * gain[c];
        sg += gc;
        sgx += gc * (xr[c] - m) * k;
    }
    sg = oann_bsum(sg);
    sgx = oann_bsum(sgx);
    real mg = rms ? (real)0 : sg / n;
    real mgx = sgx / n;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        real v = k * (gr[c] * gain[c] - mg - (xr[c] - m) * k * mgx);
        dr[c] = add ? v + dr[c] : v;
    }
}

// dgain[c] (+)= sum over rows of dy xhat, dbias[c] (+)= sum of dy; either address 0 where not wanted. Tiled as colsum: 32
// columns by 32 row lanes, the partials added in lane order
extern "C" __global__ void oann_ad_norm_param_grad(real *dgain, real *dbias, const real *dy, const real *x, const real *mean, const real *inv, long long rows, long long cols, long long addG, long long addB) {
    __shared__ real pg[32][33];
    __shared__ real pb[32][33];
    int tx = threadIdx.x & 31, ty = threadIdx.x >> 5;
    long long c = (long long)blockIdx.x * 32 + tx;
    real ag = (real)0, ab = (real)0;
    if (c < cols) {
        for (long long r = ty; r < rows; r += 32) {
            real g = dy[r * cols + c];
            ag += g * (x[r * cols + c] - mean[r]) * inv[r];
            ab += g;
        }
    }
    pg[ty][tx] = ag;
    pb[ty][tx] = ab;
    __syncthreads();
    if (ty == 0 && c < cols) {
        real tg = (real)0, tb = (real)0;
        for (int i = 0; i < 32; i++) {
            tg += pg[i][tx];
            tb += pb[i][tx];
        }
        if (dgain) dgain[c] = addG ? tg + dgain[c] : tg;
        if (dbias) dbias[c] = addB ? tb + dbias[c] : tb;
    }
}

// ---- lookups ----

extern "C" __global__ void oann_ad_embed(real *y, const int *tokens, const real *table, long long n, long long cols) {
    OANN_LOOP(i, n) y[i] = table[(long long)tokens[i / cols] * cols + i % cols];
}

// dtable (+)= the rows of dy added at their tokens' rows: one block a table row, the tokens in order, so every sum is
// in the CPU's order (and the same every run)
extern "C" __global__ void oann_ad_embed_grad(real *dtable, const real *dy, const int *tokens, long long rows, long long cols, long long add) {
    long long v = blockIdx.x;
    for (long long c = threadIdx.x; c < cols; c += blockDim.x) {
        real s = add ? dtable[v * cols + c] : (real)0;
        for (long long r = 0; r < rows; r++) {
            if (tokens[r] == v) s += dy[r * cols + c];
        }
        dtable[v * cols + c] = s;
    }
}

// y = x + the table's row (first + r % ctx) % trows
extern "C" __global__ void oann_ad_positions(real *y, const real *x, const real *table, long long n, long long cols, long long ctx, long long first, long long trows) {
    OANN_LOOP(i, n) {
        long long r = i / cols, c = i % cols;
        y[i] = x[i] + table[((first + r % ctx) % trows) * cols + c];
    }
}

// dtable (+)= dy summed over the sequences at each position's row: one thread a table element, in the CPU's order
extern "C" __global__ void oann_ad_positions_grad(real *dtable, const real *dy, long long rows, long long cols, long long ctx, long long first, long long trows, long long add) {
    OANN_LOOP(i, trows * cols) {
        long long p = i / cols, c = i % cols;
        real s = add ? dtable[i] : (real)0;
        long long t0 = ((p - first) % trows + trows) % trows;
        for (long long q = 0; q < rows / ctx; q++) {
            for (long long t = t0; t < ctx; t += trows) s += dy[(q * ctx + t) * cols + c];
        }
        dtable[i] = s;
    }
}

// ---- sums and losses ----

// y[r] = k sum of row r (k = 1 / cols for the mean)
extern "C" __global__ void oann_ad_sumlast(real *y, const real *x, long long cols, real k) {
    const real *xr = x + (long long)blockIdx.x * cols;
    real s = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) s += xr[j];
    s = oann_bsum(s);
    if (threadIdx.x == 0) y[blockIdx.x] = k * s;
}

// dx (+)= k dy[r] spread over row r
extern "C" __global__ void oann_ad_sumlast_grad(real *dx, const real *dy, long long n, long long cols, real k, long long add) {
    OANN_LOOP(i, n) { real v = k * dy[i / cols]; dx[i] = add ? dx[i] + v : v; }
}

// cross entropy, the rows: probs' row r = softmax of the logits' row r; rowloss[r] = log-sum-exp - the class's logit
// (0 for an ignored row)
extern "C" __global__ void oann_ad_ce_rows(real *probs, double *rowloss, const real *z, const int *classes, long long cols) {
    long long r = blockIdx.x;
    const real *zr = z + r * cols;
    real *pr = probs + r * cols;
    real top = zr[0];
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) top = zr[j] > top ? zr[j] : top;
    top = oann_bmax(top);
    real s = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        real e = oann_exp(zr[j] - top);
        pr[j] = e;
        s += e;
    }
    s = oann_bsum(s);
    real scale = (real)1.0 / s;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) pr[j] = pr[j] * scale;
    if (threadIdx.x == 0) {
        long long c = classes[r];
        rowloss[r] = c < 0 ? 0.0 : (double)((top + log(s)) - zr[c]);
    }
}

// the loss: the mean of rowloss over the rows counted, in double, by one block in a fixed order; stats[0] = the loss,
// stats[1] = the rows counted; value = the loss in real
extern "C" __global__ void oann_ad_ce_total(double *stats, real *value, const double *rowloss, const int *classes, long long rows) {
    double s = 0.0;
    long long counted = 0;
    for (long long r = threadIdx.x; r < rows; r += blockDim.x) {
        if (classes[r] >= 0) { s += rowloss[r]; counted++; }
    }
    s = oann_bsum(s);
    counted = (long long)oann_bsum((double)counted);
    if (threadIdx.x == 0) {
        double loss = counted > 0 ? s / (double)counted : 0.0;
        stats[0] = loss;
        stats[1] = (double)counted;
        value[0] = (real)loss;
    }
}

// dz (+)= dloss (probs - onehot(class)) / counted, zero for an ignored row; dloss = dy[0] (a device word)
extern "C" __global__ void oann_ad_ce_grad(real *dz, const real *probs, const int *classes, const real *dloss, const double *stats, long long n, long long cols, long long add) {
    real counted = (real)stats[1];
    real k = dloss[0] / (counted > (real)0 ? counted : (real)1);
    OANN_LOOP(i, n) {
        long long r = i / cols, j = i % cols;
        long long c = classes[r];
        real v = c < 0 ? (real)0 : k * (probs[i] - (j == c ? (real)1.0 : (real)0.0));
        dz[i] = add ? dz[i] + v : v;
    }
}

// the mean of (a - b)^2 in double by one block; stats[0] = it; value = it in real
extern "C" __global__ void oann_ad_mse(double *stats, real *value, const real *a, const real *b, long long n) {
    double s = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        double d = (double)(a[i] - b[i]);
        s += d * d;
    }
    s = oann_bsum(s);
    if (threadIdx.x == 0) {
        stats[0] = s / (double)n;
        stats[1] = 0.0;
        value[0] = (real)(s / (double)n);
    }
}

// da (+)= 2 dloss (a - b) / n, db (+)= the negative; an address of 0 is a gradient not wanted
extern "C" __global__ void oann_ad_mse_grad(real *da, real *db, const real *a, const real *b, const real *dloss, long long n, long long addA, long long addB) {
    real k = (real)2.0 * dloss[0] / (real)n;
    OANN_LOOP(i, n) {
        if (da) { real v = k * (a[i] - b[i]); da[i] = addA ? da[i] + v : v; }
        if (db) { real v = k * (b[i] - a[i]); db[i] = addB ? db[i] + v : v; }
    }
}

// ---- the optimizer ----

// the sum of the squares of n elements, in double, by one block in a fixed order, into *out
extern "C" __global__ void oann_ad_sumsq(double *out, const real *x, long long n) {
    double s = 0.0;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) s += (double)x[i] * (double)x[i];
    s = oann_bsum(s);
    if (threadIdx.x == 0) *out = s;
}

// x *= max / (norm + 1e-6) where norm = sqrt(*sumsq) is above max: optim.ClipNorm's rule on the device
extern "C" __global__ void oann_ad_clip(real *x, const double *sumsq, long long n, double max) {
    double norm = sqrt(*sumsq);
    if (!(norm > max)) return;
    real k = (real)(max / (norm + 1e-6));
    OANN_LOOP(i, n) x[i] = x[i] * k;
}

// nn/optim's AdamW step over n parameters, adamRun operation for operation, the decay only where decays[k] is not 0
// (decays 0: every parameter decays) - bit for bit with the CPU's
extern "C" __global__ void oann_ad_adamw(real *p, const real *g, real *m, real *v, const unsigned char *decays, long long n, real b1, real b2, real c1, real c2, const real *hp, real eps) {
    real keep = hp[0], step = hp[1], root = hp[2];   // the step's scalars, which change each step: read from the device so a CUDA graph replays them
    OANN_LOOP(k, n) {
        real gk = g[k];
        real mk = b1 * m[k] + c1 * gk;
        real vk = b2 * v[k] + c2 * gk * gk;
        m[k] = mk;
        v[k] = vk;
        real kp = (decays == 0 || decays[k]) ? keep : (real)1.0;
        p[k] = kp * p[k] - step * mk / (sqrt(vk) * root + eps);
    }
}

// ---- views (GP8): Permute, Slice and Expand read through strides, and their gradients back ----

// y (+)= x read through strides: y dense over n elements of dims (d0 implied, d1 ... d5 for the axes after the
// first, rank of them in all), element i at coordinates c read at x[sum c_k s_k]. Axis wa (-1: none) is a window
// [wlo, whi): a coordinate there is read at c - wlo, and outside it nothing is read (y written 0, or left) - a slice's
// gradient into its base's. A contiguous copy of a view (wa -1, add 0) and a permutation's gradient (the strides
// inverted) are the same kernel
extern "C" __global__ void oann_ad_gather(real *y, const real *x, long long n, long long add, long long rank, long long wa, long long wlo, long long whi,
    long long d1, long long d2, long long d3, long long d4, long long d5,
    long long s0, long long s1, long long s2, long long s3, long long s4, long long s5) {
    const long long d[6] = {1, d1, d2, d3, d4, d5};
    const long long s[6] = {s0, s1, s2, s3, s4, s5};
    OANN_LOOP(i, n) {
        long long r = i, off = 0;
        bool in = true;
        for (int k = (int)rank - 1; k >= 0; k--) {
            long long c = k > 0 ? r % d[k] : r;
            r = k > 0 ? r / d[k] : 0;
            if (k == wa) {
                if (c < wlo || c >= whi) in = false;
                c -= wlo;
            }
            off += c * s[k];
        }
        if (in) y[i] = add ? y[i] + x[off] : x[off];
        else if (!add) y[i] = (real)0;
    }
}

// an Expand's gradient: dx (+)= dy summed over what was broadcast. dx dense over n elements; the view's axes (rank of
// them, dims v) each kept - at dx's stride xs_k, its coordinate (j / xs_k) % v_k - or broadcast (xs_k 0), summed over;
// dy dense, at strides t. The sum runs over dy in its own order, as the CPU's walk adds, so the two agree bit for bit
extern "C" __global__ void oann_ad_unexpand(real *dx, const real *dy, long long n, long long add, long long rank,
    long long v0, long long v1, long long v2, long long v3, long long v4, long long v5,
    long long x0, long long x1, long long x2, long long x3, long long x4, long long x5,
    long long t0, long long t1, long long t2, long long t3, long long t4, long long t5) {
    const long long v[6] = {v0, v1, v2, v3, v4, v5};
    const long long xs[6] = {x0, x1, x2, x3, x4, x5};
    const long long t[6] = {t0, t1, t2, t3, t4, t5};
    OANN_LOOP(j, n) {
        long long base = 0, reps = 1;
        for (int k = 0; k < (int)rank; k++) {
            if (xs[k] != 0) base += ((j / xs[k]) % v[k]) * t[k];
            else reps *= v[k];
        }
        real acc = add ? dx[j] : (real)0;
        for (long long b = 0; b < reps; b++) {
            long long r = b, off = base;
            for (int k = (int)rank - 1; k >= 0; k--) {
                if (xs[k] != 0) continue;
                off += (r % v[k]) * t[k];
                r /= v[k];
            }
            acc = acc + dy[off];
        }
        dx[j] = acc;
    }
}

// ---- element functions: Shift, Sqrt, Log (ad/ops) ----

// y = x + k
extern "C" __global__ void oann_ad_shift(real *y, const real *x, real k, long long n) { OANN_LOOP(i, n) y[i] = x[i] + k; }

// y = sqrt x (NaN below 0)
extern "C" __global__ void oann_ad_sqrt(real *y, const real *x, long long n) { OANN_LOOP(i, n) y[i] = sqrt(x[i]); }

// dx (+)= dy / (2 y), read from y = sqrt x
extern "C" __global__ void oann_ad_sqrt_grad(real *dx, const real *dy, const real *y, long long n, long long add) {
    OANN_LOOP(i, n) { real v = dy[i] / ((real)2.0 * y[i]); dx[i] = add ? dx[i] + v : v; }
}

// y = log x (-infinity at 0, NaN below)
extern "C" __global__ void oann_ad_log(real *y, const real *x, long long n) { OANN_LOOP(i, n) y[i] = log(x[i]); }

// dx (+)= dy / x
extern "C" __global__ void oann_ad_log_grad(real *dx, const real *dy, const real *x, long long n, long long add) {
    OANN_LOOP(i, n) { real v = dy[i] / x[i]; dx[i] = add ? dx[i] + v : v; }
}

// ---- stablemax cross entropy (ad/ops' StablemaxCE): s(x) = x + 1 from 0 up, 1 / (1 - x) below ----

__device__ __forceinline__ real oann_stable(real x) { return x >= (real)0 ? x + (real)1.0 : (real)1.0 / ((real)1.0 - x); }

// the rows: rowloss[r] = log(sum s(z) / s(z[class])) (0 for an ignored row); the total is oann_ad_ce_total's
extern "C" __global__ void oann_ad_sm_rows(double *rowloss, const real *z, const int *classes, long long cols) {
    long long r = blockIdx.x;
    const real *zr = z + r * cols;
    real s = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) s += oann_stable(zr[j]);
    s = oann_bsum(s);
    if (threadIdx.x == 0) {
        long long c = classes[r];
        rowloss[r] = c < 0 ? 0.0 : (double)log(s / oann_stable(zr[c]));
    }
}

// dz (+)= dloss s'(z_j) (1 / sum s - [j = class] / s(z_class)) / counted, zero for an ignored row - a block a row;
// dloss = dy[0] (a device word), counted = stats[1] (oann_ad_ce_total's)
extern "C" __global__ void oann_ad_sm_grad(real *dz, const real *z, const int *classes, const real *dloss, const double *stats, long long cols, long long add) {
    long long r = blockIdx.x;
    const real *zr = z + r * cols;
    real *dr = dz + r * cols;
    long long c = classes[r];
    real counted = (real)stats[1];
    real k = dloss[0] / (counted > (real)0 ? counted : (real)1);
    real total = (real)0;
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) total += oann_stable(zr[j]);
    total = oann_bsum(total);
    for (long long j = threadIdx.x; j < cols; j += blockDim.x) {
        real s = oann_stable(zr[j]);
        real slope = s >= (real)1.0 ? (real)1.0 : s * s;
        real hit = j == c ? (real)1.0 / s : (real)0.0;
        real v = c < 0 ? (real)0 : k * slope * ((real)1.0 / total - hit);
        dr[j] = add ? dr[j] + v : v;
    }
}

// ---- rotary positions (kernels/attention's Rope, Su et al.) ----

// y (+)= x rotated, dir 1 (or back, -1: the gradient): row r at position pos + r % ctx, in each head of dh columns the
// pair (c, c + dh / 2) by the angle dir position base^(-2c / dh), the angle and its cosine and sine in double as the
// CPU's, the rotation in real; n = rows x cols / 2 pairs. y may be x where not added
extern "C" __global__ void oann_ad_rope(real *y, const real *x, long long n, long long cols, long long dh, long long ctx, long long pos, double logbase, double dir, long long add) {
    long long half = dh / 2, pairs = cols / 2;
    OANN_LOOP(i, n) {
        long long r = i / pairs, q = i % pairs;
        long long c = q % half;
        long long o = r * cols + (q / half) * dh + c;
        double angle = dir * (double)(pos + r % ctx) * exp(-2.0 * (double)c / (double)dh * logbase);
        real cs = (real)cos(angle), sn = (real)sin(angle);
        real a = x[o], b = x[o + half];
        real u = a * cs - b * sn, v = a * sn + b * cs;
        y[o] = add ? y[o] + u : u;
        y[o + half] = add ? y[o + half] + v : v;
    }
}
