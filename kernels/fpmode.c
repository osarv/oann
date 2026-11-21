/* kernels/fpmode: the x86 floating-point control bits a program sets for itself - flush-to-zero (FTZ, bit 15) and
   denormals-are-zero (DAZ, bit 6) of MXCSR. Elsewhere nothing is done. */
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

int oann_flush_denormals(int on) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned int csr = _mm_getcsr();
    if (on) csr |= 0x8040u; else csr &= ~0x8040u;
    _mm_setcsr(csr);
    return 1;
#else
    (void)on;
    return 0;
#endif
}

int oann_flushing(void) {
#if defined(__x86_64__) || defined(__i386__)
    return (_mm_getcsr() & 0x8040u) == 0x8040u;
#else
    return 0;
#endif
}
