// rate.cu - the settle's rate step on the GPU, fused (docs/redesign.md 1.10, "The first own kernel"): for every element
// i of a population's rows, as dyn/neuron's RateStep and the settle's retired units (dyn/settle's rest and retire):
//
//     r = total[i] - u[i]                      0 for a retired unit
//     word = max(word, |r|'s bits)             the residual's sup norm, kept as RateStep keeps it (below)
//     u2[i] = u[i] + dt r                      0 for a retired unit
//     s2[i] = rho(u2[i])                       0 for a retired unit
//
// one read of three arrays and one write of two. CUDA C text compiled by NVRTC when a plan is made
// (kernels/gpu/kern.olang), after elem.cu's text (oann_tanh, oann_relu, oann_clip, oann_lif_rate) and these #defines:
//   real      float or double, the state's type
//   word      unsigned or unsigned long long, an unsigned integer of real's width
//   OANN_RHO  the activation of a real x, an expression: x, oann_tanh(x), oann_clip(x), oann_relu(x), or LifRate's
//             (real)oann_lif_rate((double)x, threshold, tau) with its numbers written in
//
// The sup norm as bits: |r|'s pattern (the sign cleared) orders as an unsigned integer as |r| does, and every NaN is
// above the infinities, so an integer maximum is the largest |r|, a NaN where one is - exact and order-free, so one
// atomicMax per block into the word gives the CPU's sup norm whatever the order the threads run in. The word is a
// maximum over everything launched into it since it was zeroed: several populations' steps of one sweep may share it.
//
// dt is per phase: dts[i / per], per the elements of a phase's rows (batch x size; the rows are phase-major). Indices
// are 32-bit (kern.olang refuses more than 2^31 - 1 elements). Arithmetic is not contracted into fused multiply-adds
// (--fmad=false), so u2 is the CPU's bit for bit.

__device__ __forceinline__ word oann_bits(float x) { return __float_as_uint(x); }
__device__ __forceinline__ word oann_bits(double x) { return (word)__double_as_longlong(x); }

// the largest of every thread's m in the block into *out by one atomicMax (block a multiple of 32, at most 1024)
__device__ __forceinline__ void oann_max_into(word *out, word m) {
    __shared__ word part[32];
    for (int o = 16; o > 0; o >>= 1) {
        word other = __shfl_down_sync(0xffffffffu, m, o);
        m = other > m ? other : m;
    }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) part[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = lane < (int)(blockDim.x >> 5) ? part[lane] : (word)0;
        for (int o = 16; o > 0; o >>= 1) {
            word other = __shfl_down_sync(0xffffffffu, m, o);
            m = other > m ? other : m;
        }
        if (lane == 0 && m != 0) atomicMax(out, m);
    }
}

// the fused step over elements [0, n) of the arrays given (each already at its row offset); alive, where not null,
// says per unit (i % size) whether it is not retired
extern "C" __global__ void oann_rate_step(const real *u, const real *total, real *u2, real *s2, const real *dts,
                                          unsigned per, const unsigned char *alive, unsigned size, unsigned n,
                                          word *out) {
    word m = 0;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        real ui = u[i];
        real r = total[i] - ui;
        real v = ui + dts[i / per] * r;
        real s = OANN_RHO(v);
        if (alive != 0 && alive[i % size] == 0) r = v = s = real(0);
        word b = oann_bits(r) << 1 >> 1;
        m = b > m ? b : m;
        u2[i] = v;
        s2[i] = s;
    }
    oann_max_into(out, m);
}

// ---- the same step unfused, three launches: what the fused one is measured against (G-M3, bench/gpurate.olang) ----

// r[i] = total[i] - u[i], its sup norm into *out
extern "C" __global__ void oann_rate_residual(const real *u, const real *total, real *r, const unsigned char *alive,
                                              unsigned size, unsigned n, word *out) {
    word m = 0;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        real ri = total[i] - u[i];
        if (alive != 0 && alive[i % size] == 0) ri = real(0);
        word b = oann_bits(ri) << 1 >> 1;
        m = b > m ? b : m;
        r[i] = ri;
    }
    oann_max_into(out, m);
}

// u2[i] = u[i] + dt r[i]
extern "C" __global__ void oann_rate_move(const real *u, const real *r, real *u2, const real *dts, unsigned per,
                                          const unsigned char *alive, unsigned size, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        real v = u[i] + dts[i / per] * r[i];
        if (alive != 0 && alive[i % size] == 0) v = real(0);
        u2[i] = v;
    }
}

// s2[i] = rho(u2[i])
extern "C" __global__ void oann_rate_act(const real *u2, real *s2, unsigned n) {
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        real v = u2[i];
        s2[i] = OANN_RHO(v);
    }
}
