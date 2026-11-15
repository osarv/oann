# kernels

The numeric kernels every layer of oann runs on: functions over 2-D strided operands (`Mat<S>`) and raw arrays,
independent of `Tensor`, generic over the storage type `S` and the accumulator `A`. They check nothing per element and
no shapes either: a caller checks a shape once, where it is given (docs/redesign.md, section 3). Each kernel splits its
output over tasks (`join`/`spawn`), so a result is the same for every thread count.

| file | what |
|---|---|
| `mat.olang` | `Mat<S>`, the operand - Rows x Cols elements of an array from `Off`, rows `Ld` apart; `Act`, the activations elem and gemm's epilogue share |
| `gemm.olang` | `Gemm` (Goto's blocked GEMM, the target's tile, an epilogue of bias and activation; a storage type per operand), `GemmBatch` (attention's per-head products, causal triangles skipped), `GemmLeft` (a left operand packed by its own type: a convolution's patches), `Gemv` (one vector, no packing - the settle's blocks), `Ger2`, `GemmI8` (U8 x I8 in I32, VNNI's `vpdpwssd`) and `GemmI8Left` (its activations packed by their own type: an INT8 convolution's windows), `Workspace<A>` (the packing panels and the thread count, reserved: `Reserve`, `ReserveLayer`, `ReserveI8`) |
| `scan.olang` | linear recurrences, each a sequential scan and a chunked form, forward and backward: `GatedScan` (h = a h + b x per channel: the consolidator's slow part, QRNN, discretized state-space models), `SelectiveScan` (Mamba's: N states per channel, input-dependent B, C and step, readout and skip in the pass), `DeltaRule` (DeltaNet's S = alpha S (I - beta k k^T) + beta v k^T and Gated DeltaNet's gate; chunked in the WY form, a chunk's work as gemm's products); states in the accumulator, passed in and out, a backward from the states the forward kept per chunk; `bench/scan.olang` times the two forms |
| `sparse.olang` | `Csr` (compressed sparse rows), `Tiles` (a mask of tiles, only live ones stored), `Events` (the units that changed by more than a threshold) and their products with a batch - `MulAdd` forward and transposed, `Sampled` at the entries, `Transport` of the events' rows - `RowTopK`/`TopK`, the sparse-code gather and scatter, `Macs` (dense against effective multiply-adds); pruning in place (`Mask`, `Drop`, `Squeeze`, `Compact`) without reallocating |
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

Tests are in each file (`olang test kernels/gemm.olang`); scan's hold the chunked forms to the sequential ones to 1e-5 in F32 and every backward to central differences in F64. Speed against linalg's product, `Gemv` on a settle's blocks and `GemmI8` against quant's product were measured before C3 removed those modules (M4, M5, M12 of docs/redesign.md 4.3; the driver is `bench/gemm.olang` at branch `pre-c3`).

Sparse and event-driven (`sparse.olang`, `conv.olang`): weights are stored in S and widened as they are read; batches (feature-major: row j
holds feature j of every sample), results and gradients are in the accumulator A. A pattern a caller gives is checked once, where the matrix is
made (`PatternError`, `GeometryError`). A product given `threads` splits its output over tasks and is the same for every count; given a `Macs` it adds
the multiply-adds a dense product would have done and those it did. Pruning at run time: `Mask` zeroes entries in place, `Drop` removes them in
place (a `Tiles` retires whole tiles, `Compact` moves the live ones together), `Squeeze` moves a companion array (gradient, moments) the same way.
Speed against dense and against the old modules was measured before C3 removed them (M10 of docs/redesign.md 4.3; `bench/sparse_kernels.olang` at branch `pre-c3`).
