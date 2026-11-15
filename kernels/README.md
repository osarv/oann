# kernels

The numeric kernels every layer of oann runs on: functions over 2-D strided operands (`Mat<S>`) and raw arrays,
independent of `Tensor`, generic over the storage type `S` and the accumulator `A`. They check nothing per element and
no shapes either: a caller checks a shape once, where it is given (docs/redesign.md, section 3). Each kernel splits its
output over tasks (`join`/`spawn`), so a result is the same for every thread count.

| file | what |
|---|---|
| `mat.olang` | `Mat<S>`, the operand - Rows x Cols elements of an array from `Off`, rows `Ld` apart; `Act`, the activations elem and gemm's epilogue share |
| `gemm.olang` | `Gemm` (Goto's blocked GEMM, the target's tile, an epilogue of bias and activation), `GemmBatch` (attention's per-head products, causal triangles skipped), `GemmLeft` (a left operand packed by its own type: a convolution's patches), `Gemv` (one vector, no packing - the settle's blocks), `Ger2`, `GemmI8` (U8 x I8 in I32, VNNI's `vpdpwssd`), `Workspace<A>` (the packing panels and the thread count) |
| `scan.olang` | linear recurrences, each a sequential scan and a chunked form, forward and backward: `GatedScan` (h = a h + b x per channel: the consolidator's slow part, QRNN, discretized state-space models), `SelectiveScan` (Mamba's: N states per channel, input-dependent B, C and step, readout and skip in the pass), `DeltaRule` (DeltaNet's S = alpha S (I - beta k k^T) + beta v k^T and Gated DeltaNet's gate; chunked in the WY form, a chunk's work as gemm's products); states in the accumulator, passed in and out, a backward from the states the forward kept per chunk; `bench/scan.olang` times the two forms |

Storage and accumulation: F32 x F32, F64 x F64, and BF16, F16, F8E4M3, F8E5M2 stored and summed in F32 - widened as
operands are packed, rounded to `S` once as each element is stored, also over a depth of several blocks; integers sum
in themselves (I32), and `GemmI8` sums U8 activations times I8 weights in I32.

Tests are in each file (`olang test kernels/gemm.olang`); scan's hold the chunked forms to the sequential ones to 1e-5 in F32 and every backward to central differences in F64. Speed: `bench/gemm.olang` measures `Gemm` against linalg's
product on the networks' shapes (F32 and BF16), `Gemv` on a settle's blocks, and `GemmI8` against quant's product -
`olang build -output build/bench_gemm bench/gemm.olang && build/bench_gemm ROUNDS THREADS`.
