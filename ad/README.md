# ad

Reverse-mode autodiff on N-d tensors: the recorded graph oann's models train on.

| file | what |
|---|---|
| `graph.olang` | `Graph<T>`, `Var`, `Op` (a closed enum, `Custom(any Function&)` for operations defined outside), `Init`, `Mode`, `Function`. Recording with shape checks (a mismatch is reported by `Plan` as a `ShapeError`), the plan - parameters, their gradients, inputs, then one activation region in which every value, gradient, kept tensor and scratch is placed by its lifetime over a step - and Forward/Backward dispatch; eager mode (each op runs as recorded, on its own storage); views (`Reshape`, `Permute`, `Slice`, `Expand`) that copy nothing; BF16/F16 graphs with F32 masters; KV-cache decoding; safetensors checkpoints (io/checkpoint); INT8 inference of the Linear layers (kernels/gemm's `GemmI8`); profiling by op kind |
| `ops.olang` | every operation's forward and vector-Jacobian product over the kernels (kernels/gemm, elem, attention, scan, conv): products and linear layers, element-wise ops and activations, `Sqrt`, `Log`, scalar operands, softmax, log-softmax, cross-entropy, stablemax cross-entropy, MSE, layer and RMS norm, embedding, learned and rotary positions, attention, dropout, convolution and pooling, the gated and selective scans and the delta rule, reductions, the views' backward |

Rules the graph keeps: a step (Forward, Backward) allocates nothing - its resident memory is flat after the first
(tested); eager and captured runs agree bit for bit; results are the same for every thread count; an intermediate's
value or gradient is valid within its step only (the activation region is reused), while inputs, parameters, outputs
(nodes nothing reads), losses and nodes given to `Keep` live to the step's end.

Element functions and scalars: `g.Sqrt(x)`, `g.Log(x)` (NaN below 0; `Sqrt`'s gradient is infinite at 0, `Log`'s -infinity value at 0), and a
constant against every element - `AddScalar`, `SubScalar`, `MulScalar`, `DivScalar`, `ScalarSub(k, x)` (k - x). `g.StablemaxCE(logits, classes)` is
`CrossEntropy` over stablemax (s(x) = x + 1 for x >= 0, 1 / (1 - x) below, normalized over the last axis), for low
precision where softmax saturates; ignored classes (negative) as `CrossEntropy`.

Evaluation without gradients: set `g.NoGrad = true` before `Plan`. The graph then plans no gradient buffers (a smaller
arena) and `Backward` is refused; Forward is the same numbers.

One parameter set in two graphs (a model and a copy of it on other inputs, a target network): record both the same
way, `b.CopyParamsFrom(a)` before each of b's Forward (after a's optimizer step), run both Backwards, then
`a.AddGradsFrom(b)` - a's `ParamGrads()` is then the gradient of the sum of the two losses, which a's optimizer steps.
(Each graph keeps its own storage; the two calls are the sharing.)

Tests: `olang test ad/graph.olang` (the graph's own), `olang test tools/gradcheck.olang` (a gradient check per operation
in F64). Measures: `bench/mlp_ad.olang` (MNIST MLP), `bench/arena_ad.olang` (the transformer's arena and
resident memory per step), `bench/smallgraph.olang` (a small recurrence where the cost per node dominates) - `make mlpad`,
`make arenaad`.
