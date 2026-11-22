# ad

Reverse-mode autodiff on N-d tensors: the recorded graph oann's models train on.

| file | what |
|---|---|
| `graph.olang` | `Graph<T>`, `Var`, `Op` (a closed enum, `Custom(any Function&)` for operations defined outside), `Init`, `Mode`, `Function`. Recording with shape checks (a mismatch is reported by `Plan` as a `ShapeError`), the plan - parameters, their gradients, inputs, then one activation region in which every value, gradient, kept tensor and scratch is placed by its lifetime over a step - and Forward/Backward dispatch; eager mode (each op runs as recorded, on its own storage); views (`Reshape`, `Permute`, `Slice`, `Expand`) that copy nothing; BF16/F16 graphs with F32 masters; KV-cache decoding; attention over a context of `FlashFrom` (512) or more keeps no weights on the CPU (kernels/attention's flash form); safetensors checkpoints (io/checkpoint); INT8 inference of the Linear layers (kernels/gemm's `GemmI8`); profiling by op kind |
| `descent.olang` | `Descent<T>`: a state settled by gradient descent on an energy recorded on the graph (`Settle`, with Langevin noise by `Temperature` and `Cooling`, a low-bit state by `StateBits` (or `ReadRounded`), deep supervision of Unrolled by `Ds`; nudged by `beta` toward a cost), and the parameters' gradient of the cost at the fixed point three ways (`Train` by a `Trainer`): `Unrolled(h)` (backpropagation through the last h steps), `Implicit(tol, iterations)` (the exact gradient: the adjoint solved by conjugate gradients), `Ep(beta, centered)` (equilibrium propagation: free and nudged settles, the contrast of dE/dtheta), and a contrastive trainer from two starts, `Margin(m)` (settled from where the state is and from `SetTarget`'s target; where a row's two fixed points differ and its energies are less than m apart, the hinge's gradient - the target's basin pushed down, the other pushed up; the energy row by row in `RowEnergy`, the rows that stepped in `Active`), and the same contrast between two settled states as a term added to any trainer's gradient, `AddMargin(m, weight, start, right)` (a second settle from start, the caller's `right` judging each row's two settled states) |
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

Energy descent (one network, either training): record the state as an `Input` given to `Watch`, the energy E(x, theta)
and a cost C(x) (reading the state and data, not the parameters) as scalar nodes - sum over the rows twice,
`g.Sum(g.Sum(v))` - then `Plan` and make `descent.Descent<F32>(g, x, e, c)` (set `Rate` below 2 / E's largest
curvature in x, `Tol`, `Steps`; `Eps`, the central differences' step, about 1e-2 for F32). Fill the state with the
settle's start, then `d.Train(descent.Trainer.Ep(0.05, true))` (or `Unrolled(h)`, `Implicit(tol, iterations)`):
it settles, leaves the state at the free fixed point (C's value the free cost) and dL/dtheta in `ParamGrads()`, which
an optimizer then steps as after `Backward`. Each step is one `Forward` and `Backward` from E (and from C when nudged);
`Unrolled` and `Implicit` take the second derivatives they need - H v and (d dE/dx / dtheta)^T v - as central
differences of the first along v, two passes each, since the graph records no second-order backward; EP needs none.

Score matching (`score.olang`): `ScoreMatch(g, z, energy, clean, sigma, r, ...)` trains an energy E(z) written on
the graph so that its gradient in z matches the denoising score (Vincent 2011): z = clean + sigma eps, the loss
sum |sigma dE/dz - eps|^2 per row, and its parameter gradient - which needs d(dE/dz)/dtheta - as a central difference
of dE/dtheta along the residual, two passes, no second-order backward. The energy then ranks candidates (lower is
more plausible) or drives Langevin descent.

Two Backwards after one Forward (two losses of one graph): run the one from the node recorded later first. The plan
places buffers by a backward from the last node, so the earlier loss's gradients may overwrite a value the later one's
backward still reads.

Tests: `olang test ad/graph.olang`, `olang test ad/descent.olang`, `olang test ad/score.olang` (the graph's own), `olang test tools/gradcheck.olang` (a gradient check per operation
in F64). Measures: `bench/mlp_ad.olang` (MNIST MLP), `bench/arena_ad.olang` (the transformer's arena and
resident memory per step), `bench/smallgraph.olang` (a small recurrence where the cost per node dominates) - `make mlpad`,
`make arenaad`.

A settle's steps need only the state's gradient: `Backward(loss, accumulate, params = false)` skips every parameter's
gradient (the weights' products and column sums) and leaves `ParamGrads` as it was; `descent.Descent`'s settles step this
way and take dE/dtheta once, at the fixed point a trainer needs it.
