// settle.cu - dyn's settle on the GPU (docs/redesign.md 1.10, "The settle on the GPU"; dyn/mirror.olang, GP4): the
// sweep's step for one population's rows with its constant, its carries and a readout's nudge fused in, the stop that
// keeps the state a settle converged on, the rows' Euclidean residual, an input's closed form, the constants, and EP's
// differences and bias sums. CUDA C text compiled by NVRTC when a plan is made, after elem.cu's text (oann_exp,
// oann_tanh, oann_relu, oann_clip, oann_lif_rate) and these #defines (dyn/mirror.olang writes them per population):
//   real        float or double, the state's type
//   OANN_RHO    the activation of a real x (kern.olang's rhoText)
//   OANN_FORCE  0: none; 1: a squared readout's, target - s; 2: a cross-entropy readout's, onehot(y) - softmax(s / T)
//   OANN_INV    the cross-entropy's 1 / T as a real (its bits)
//
// The residual words. A settle has one 64-bit word per sweep, zeroed at its start. In the sup norm (the default) a
// step puts the largest |r| of its rows into its sweep's word as bits - |r|'s pattern, the sign cleared, widened from
// a float's 32 bits - by one atomicMax a block, so the populations of a sweep share it and the order of the threads does
// not matter (rate.cu's argument). Under a Euclidean budget the steps' sup norm goes into a word of its own and
// oann_rows_residual writes the sweep's word as the largest row's Euclidean norm, a double's bits.
//
// The stop. Each step reads, before anything else, the stop word and the word of the sweep before its own: once a
// sweep's residual is within the tolerance (its bits at most tol, the largest float at most the tolerance) or not
// finite (at least inf's bits), the next sweep's steps record that sweep in the stop word and write nothing, nor does
// any later one. So the state the converged sweep read stays in its buffer - the state dyn/settle's relax keeps - and
// the host, reading the words every few sweeps, finds the first one that stopped. Arithmetic is not contracted
// (--fmad=false): a step is dyn/neuron's RateStep and AddForce operation for operation.

#define OANN_NONE 0xffffffffffffffffull

__device__ __forceinline__ unsigned long long oann_abs_bits(float x) { return (unsigned long long)(__float_as_uint(x) & 0x7fffffffu); }
__device__ __forceinline__ unsigned long long oann_abs_bits(double x) { return ((unsigned long long)__double_as_longlong(x)) & 0x7fffffffffffffffull; }

// IEEE 754's maximum, as olang's Max: a NaN wins, +0 above -0
__device__ __forceinline__ real oann_fmax(real a, real b) {
    if (a != a) return a;
    if (b != b) return b;
    if (a == b) return signbit(a) ? b : a;
    return a > b ? a : b;
}

// the largest of every thread's m in the block into *out by one atomicMax (blocks of a multiple of 32, at most 1024)
__device__ __forceinline__ void oann_max64(unsigned long long *out, unsigned long long m) {
    __shared__ unsigned long long part[32];
    for (int o = 16; o > 0; o >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffffu, m, o);
        m = other > m ? other : m;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < (int)(blockDim.x >> 5) ? part[lane] : 0ull;
        for (int o = 16; o > 0; o >>= 1) {
            unsigned long long other = __shfl_down_sync(0xffffffffu, m, o);
            m = other > m ? other : m;
        }
        if (lane == 0 && m != 0) atomicMax(out, m);
    }
}

// whether this sweep runs: no sweep stopped yet, and the sweep before (prev, its word; 0 for the first sweep) is neither
// within the tolerance nor diverged - where it is, it is recorded as the one that stopped (sweep - 1)
__device__ __forceinline__ bool oann_running(unsigned long long *stop, const unsigned long long *prev, unsigned sweep,
                                             unsigned long long tol, unsigned long long inf) {
    if (*(volatile unsigned long long *)stop != OANN_NONE) return false;
    if (prev == 0) return true;
    unsigned long long w = *prev;
    if (w <= tol || w >= inf) {
        *stop = (unsigned long long)(sweep - 1);
        return false;
    }
    return true;
}

// one sweep's step of n elements of a population's rows (each array already at its first row; the rows phase-major,
// per = batch x size elements a phase, the first a phase's start): the totals - the constant of the row's batch row,
// plus the carries where there are any (total, 0 for none), plus a readout's nudge where its phase's beta is not 0 -
// then r = totals - u, |r| into the word at out, u2 = u + dt r, s2 = rho(u2), retired units (alive 0) put at 0. dts and
// betas are the phases', s the population's activities the sweep reads (the nudge's), labels and target its batch rows'
extern "C" __global__ void oann_settle_step(const real *u, const real *total, const real *cst, real *u2, real *s2,
                                            const real *dts, unsigned per, const unsigned char *alive, unsigned size,
                                            unsigned n, unsigned long long *out, unsigned long long *stop,
                                            const unsigned long long *prev, unsigned sweep, unsigned long long tol,
                                            unsigned long long inf, const real *s, const int *labels,
                                            const real *target, const real *betas, unsigned groups) {
    if (!oann_running(stop, prev, sweep, tol, inf)) return;
    unsigned long long m = 0;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        unsigned b = i % per;
        real t = cst[b];
        if (total != 0) t = t + total[i];
#if OANN_FORCE != 0
        real k = betas[i / per];
        if (k != real(0)) {
            unsigned j = i % size, row = i / size;
#if OANN_FORCE == 1
            t = t + k * (target[b] - s[i]);
#else
            unsigned w = size / groups, g = j / w, at = row * size + g * w;
            int y = labels[(b / size) * groups + g];
            real top = s[at];
            for (unsigned q = 1; q < w; q++) top = oann_fmax(top, s[at + q]);
            real sum = real(0);
            for (unsigned q = 0; q < w; q++) sum += oann_exp((s[at + q] - top) * OANN_INV);
            real scale = real(1) / sum;
            real p = oann_exp((s[i] - top) * OANN_INV) * scale;
            real hit = (int)(j - g * w) == y ? real(1) : real(0);
            t = t + k * (hit - p);
#endif
        }
#endif
        real ui = u[i];
        real r = t - ui;
        real v = ui + dts[i / per] * r;
        real a = OANN_RHO(v);
        if (alive != 0 && alive[i % size] == 0) r = v = a = real(0);
        unsigned long long bits = oann_abs_bits(r);
        m = bits > m ? bits : m;
        u2[i] = v;
        s2[i] = a;
    }
    oann_max64(out, m);
}

// the rows' Euclidean residual of a sweep, dyn/settle's rowsResidual: for each of n rows, the sum over the relaxing
// populations (in their order) of the sum over the row's live units of ((u2 - u) inv)^2 in double, inv one over the
// row's phase's step, its square root - the largest into the word at out as a double's bits, or the sweep's sup norm
// (the word at sup, a real's bits) where that is a NaN, as relax keeps it. The populations are a table of pops entries
// of five 64-bit slots - u's address, u2's (each at the population's first row), the units, the alive mask's address
// (0: none), unused - the rows n from row r0, and invs is per phase, from the settle's first
extern "C" __global__ void oann_rows_residual(const unsigned long long *table, unsigned pops, unsigned n, unsigned batch,
                                              const double *invs, const unsigned long long *sup, unsigned long long nan,
                                              unsigned long long *out, unsigned long long *stop, unsigned r0) {
    if (*(volatile unsigned long long *)stop != OANN_NONE) return;
    if (*sup > nan) {
        if (blockIdx.x == 0 && threadIdx.x == 0) atomicMax(out, 0x7ff8000000000000ull);
        return;
    }
    unsigned long long m = 0;
    for (unsigned r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += gridDim.x * blockDim.x) {
        double inv = invs[r / batch], sum = 0.0;
        for (unsigned q = 0; q < pops; q++) {
            const real *u = (const real *)table[5 * q];
            const real *u2 = (const real *)table[5 * q + 1];
            unsigned size = (unsigned)table[5 * q + 2];
            const unsigned char *alive = (const unsigned char *)table[5 * q + 3];
            unsigned long long at = (unsigned long long)(r0 + r) * size;
            double ps = 0.0;
            for (unsigned j = 0; j < size; j++) {
                if (alive != 0 && alive[j] == 0) continue;
                double d = (double)(u2[at + j] - u[at + j]) * inv;
                ps += d * d;
            }
            sum += ps;
        }
        unsigned long long bits = oann_abs_bits(sqrt(sum));
        m = bits > m ? bits : m;
    }
    oann_max64(out, m);
}

// an input's closed form over phases x per elements: u = its drive (per elements, the batch's), s = rho(u)
extern "C" __global__ void oann_closed_form(real *u, real *s, const real *drive, unsigned per, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        real v = drive[i % per];
        u[i] = v;
        s[i] = OANN_RHO(v);
    }
}

// a population's constant before its folded carries: cst = drive + bias, n = batch x size elements
extern "C" __global__ void oann_constant(real *cst, const real *drive, const real *bias, unsigned size, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        cst[i] = drive[i] + bias[i % size];
}

// EP's differences, dyn/learn's differ: d = plus - minus over n elements
extern "C" __global__ void oann_differ(real *d, const real *plus, const real *minus, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) d[i] = plus[i] - minus[i];
}

// a bias's part of EP's estimate, dyn/learn's biasSums: out[j] = k times the sum over the rows of d[row size + j], the
// rows summed in order from zero (one thread a unit)
extern "C" __global__ void oann_bias_sums(real *out, const real *d, unsigned rows, unsigned size, real k) {
    for (unsigned j = blockIdx.x * blockDim.x + threadIdx.x; j < size; j += gridDim.x * blockDim.x) {
        real acc = real(0);
        for (unsigned r = 0; r < rows; r++) acc += d[r * size + j];
        out[j] = k * acc;
    }
}
