# kernels

The numeric kernels every layer of oann runs on: functions over 2-D strided operands (`Mat<S>`) and raw arrays,
independent of `Tensor`, generic over the storage type `S` and the accumulator `A`. They check nothing per element and
no shapes either: a caller checks a shape once, where it is given. Each kernel splits its
output over tasks (`join`/`spawn`), so a result is the same for every thread count.

| file | what |
|---|---|
| `mat.olang` | `Mat<S>`, the operand - Rows x Cols elements of an array from `Off`, rows `Ld` apart; `Act`, the activations elem and gemm's epilogue share |
| `gemm.olang` | `Gemm` (Goto's blocked GEMM, the target's tile, an epilogue of bias and activation; a storage type per operand), `GemmBatch` (attention's per-head products, causal triangles skipped), `GemmLeft` (a left operand packed by its own type: a convolution's patches), `GemmPacked` (op(B) packed once into a `Packed<A>` and read many times: a settle's weights over its sweeps), `Gemv` (one vector, no packing - the settle's blocks), `Ger2`, `GemmI8` (U8 x I8 in I32, VNNI's `vpdpwssd`) and `GemmI8Left` (its activations packed by their own type: an INT8 convolution's windows), `Workspace<A>` (the packing panels and the thread count, reserved: `Reserve`, `ReserveLayer`, `ReserveI8`) |
| `attention.olang` | multi-head scaled dot-product attention over `Mat` views of the heads: `Attention` and `AttentionGrad` (the weights made and kept, Ctx x Ctx a sequence and head, gemm's batched products), `FlashAttention` and `FlashAttentionGrad` (the weights made a block at a time and never kept - an online softmax, the backward recomputing them from each row's log-sum-exp; `Flash<A>` the tasks' workspaces), `AttentionStep` (decoding against a key/value cache), `Rope`; `bench/flash.olang` times the two forms |
| `fpmode.olang` | how the CPU treats subnormal floats: `FlushDenormals` (x86's FTZ and DAZ through `fpmode.c`), `Flushing`; `Startup`, run by `Graph.Plan` and `System.Plan`, flushes them unless `KeepSubnormals` is set first |
| `ternary.olang` | ternary weights (-1, 0, +1 times a scale per row, BitNet b1.58's) as two bit-planes, 2 bits a weight: `Ternary` (`Quantize` by absmean, `Get`, `Set`), `MatVec` and `MatMul` - sums of +x, -x or nothing, no multiplication by a weight; `bench/ternary.olang` times it against F32 |
| `spike.olang` | integer-only spiking kernels, each a datapath's one step, widths by saturation: fixed point (`Fix`, `Real`, `Sat`, `Shr` a rounding shift, `MulShr`, `FracFor`, `Quant`), `SigmaDelta` (a per-unit encoder sending signed quanta of change), `Csr` and `Scatter` (an event's integer weights times its count into each target's accumulator), `ExpLut` and `Softmax` (exp from a table, one division), `Leak`, `Lfsr` and `Gauss4` (on-chip noise) |
| `scan.olang` | linear recurrences, each a sequential scan and a chunked form, forward and backward: `GatedScan` (h = a h + b x per channel: the consolidator's slow part, QRNN, discretized state-space models), `SelectiveScan` (Mamba's: N states per channel, input-dependent B, C and step, readout and skip in the pass), `DeltaRule` (DeltaNet's S = alpha S (I - beta k k^T) + beta v k^T and Gated DeltaNet's gate; chunked in the WY form, a chunk's work as gemm's products); states in the accumulator, passed in and out, a backward from the states the forward kept per chunk; `bench/scan.olang` times the two forms |
| `sparse.olang` | `Csr` (compressed sparse rows), `Tiles` (a mask of tiles, only live ones stored), `Events` (the units that changed by more than a threshold) and their products with a batch - `MulAdd` forward and transposed, `Sampled` at the entries, `Transport` of the events' rows - `RowTopK`/`TopK`, the sparse-code gather and scatter, `Macs` (dense against effective multiply-adds); pruning in place (`Mask`, `Drop`, `Squeeze`, `Compact`) without reallocating |
| `device.olang` | where a plan runs, chosen once (docs/redesign.md 1.10): `Backend` (`Cpu`, `Gpu`), `SetDefault`, `Backend.Default()` (what a plan's `on` is by default), `GpuAvailable()`, `DeviceError`; with `-define Gpu=true` the program's device, opened by the first plan on the GPU (`Open`, `Opened`). Imports with or without the define |
| `gpu/` | the GPU backend (docs/redesign.md 1.10), built only with `-define Gpu=true` and tested by `make gputest`: `cuda.c` (the C shim over CUDA, cuBLAS and cuBLASLt; device addresses as U64), `device.olang` (`Device`, `Buf<S>`, `DMat<S>`, `GpuError`, uploads and downloads), `blas.olang` (`Gemm` by cublasGemmEx, row-major as here; `Product` and `ProductI8`, cuBLASLt's products planned once - epilogues, per-operand storage, batches over `DBatch`, INT8, exact or fast sums, tuned - all checked against kernels/gemm), `kern.olang` (our own kernels: CUDA C in `gpu/src/*.cu` compiled by NVRTC at a plan, the PTX cached by hash, launched through the driver API; `Axpy`, the settle's fused rate step `Rate`, checked against `dyn/neuron.RateStep` in `dyn/gpu_test.olang`, `AdamW`; `src/settle.cu` is dyn's settle on the GPU, `dyn/mirror.olang`); `bench/gpugemm.olang` and `bench/gpult.olang` time them (G-M1), `bench/gpurate.olang` the rate step and a sweep (G-M3) |
| `conv.olang` | `Shape`, `Conv2d` (the patch matrix never made: `Patches` is `GemmLeft`'s left operand, its padding a value of its own), `Conv2dBackward`, `ReserveConv`, `Im2col`/`Col2im`, `MaxPool2d` and its backward, over `Mat` views, channels last |

Storage and accumulation: F32 x F32, F64 x F64, and BF16, F16, F8E4M3, F8E5M2 stored and summed in F32 - widened as
operands are packed, rounded to `S` once as each element is stored, also over a depth of several blocks; integers sum
in themselves (I32), and `GemmI8` sums U8 activations times I8 weights in I32. `Gemm`, `GemmLeft` and `GemmBatch` take a
storage type per operand (`Gemm<SC, SA, SB, A>`): BF16, F16, F8 or INT8 weights under F32 activations give the product
of the weights widened first, bit for bit (tests); `bench/storage.olang` times them against F32 weights.

Workspaces are reserved, never grown: a product, an attention, a convolution or a scan takes its panels and scratch from
a `Workspace` (`Reserve(dim, wide)`, `ReserveLayer(x, y, z, like)`, `ReserveI8(n, k)`, `ReserveAttention`,
`ReserveConv`) or a `scan.Work` (`ReserveGated`, `ReserveSelective`, `ReserveDelta`) given room where the products are
planned - `ad/graph`'s `Plan`, `dyn`'s `Ready` - and one with too little ends the program naming the Reserve it needed.
So nothing here allocates, and that is pinned while compiling (`assert not Gemm<F32, F32, F32, F32>.Allocates`, S17a):
every product, attention forward and backward, convolution and sequential scan.

Tests are in each file (`olang test kernels/gemm.olang`); scan's hold the chunked forms to the sequential ones to 1e-5 in F32 and every backward to central differences in F64.

Sparse and event-driven (`sparse.olang`, `conv.olang`): weights are stored in S and widened as they are read; batches (feature-major: row j
holds feature j of every sample), results and gradients are in the accumulator A. A pattern a caller gives is checked once, where the matrix is
made (`PatternError`, `GeometryError`). A product given `threads` splits its output over tasks and is the same for every count; given a `Macs` it adds
the multiply-adds a dense product would have done and those it did. Pruning at run time: `Mask` zeroes entries in place, `Drop` removes them in
place (a `Tiles` retires whole tiles, `Compact` moves the live ones together), `Squeeze` moves a companion array (gradient, moments) the same way.
