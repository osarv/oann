# The Vulkan backend

oann's second GPU backend, for every GPU that is not NVIDIA's (and NVIDIA's too): `kernels/vk`. The first backend,
`kernels/gpu`, is CUDA with cuBLAS and cuBLASLt and runs on NVIDIA alone (docs/redesign.md 1.10).

## Why Vulkan compute

| route | reaches | needs on the user's machine |
|---|---|---|
| Vulkan compute | AMD, Intel, NVIDIA, integrated GPUs, Android phones, software (lavapipe); Apple only through MoltenVK | the GPU driver, which carries Vulkan |
| ROCm / HIP | AMD's discrete cards, Linux mostly | AMD's toolkit |
| Level Zero / SYCL | Intel | Intel's runtime |
| OpenCL | most desktops, poorly on phones | an OpenCL driver, often absent |
| Metal | Apple | nothing |

One backend instead of one a vendor, no toolkit to install, and the integrated GPUs of ordinary laptops included.
Apple is left for a Metal backend of its own later (MoltenVK is third-party code). CUDA stays for NVIDIA, where cuBLAS
is the faster product.

## How it is built

- **The binding** (`vk.c`). Vulkan's interface is structs of pointers to structs, which olang's C imports hold only as
  pointers of no type, so one C shim fills them, as `cuda.c` does for CUDA; everything crosses as numbers. The shim
  opens the Vulkan loader (`libvulkan.so.1`, `vulkan-1.dll`, Android's `libvulkan.so`) when a device is asked for and
  finds each function by name: a build needs the headers alone, links nothing, and the program runs on a machine with
  no Vulkan (`Device.Open` gives `NoDevice`).
- **Kernels are written in olang** (`spirv.olang`). Every Vulkan driver takes SPIR-V. `spirv.Kernel` records a compute
  shader as it is written - buffers, parameters, arithmetic, variables, structured `If` and `Loop` - and `Finish`
  assembles SPIR-V 1.0 (Vulkan 1.0, so every driver). No shader compiler, no GLSL, no third-party code. Floating-point
  `+ - *` are marked NoContraction: no driver fuses a multiply-add, as olang does not.
- **One queue, in program order** (`vk.c`). Uploads, kernels and downloads are recorded into one command buffer, each
  behind a barrier on what came before, and submitted at `Sync` (or sooner where staging memory or the descriptor pool
  fills). Nothing races by construction, as on CUDA's one compute stream. Buffers are device-local; transfers go
  through 64 MiB of host-visible staging memory.
- **Operands** are whole buffers bound to a kernel, their offsets and strides given as parameters (push constants), so
  a view at any offset is arithmetic and no driver's offset alignment matters.
- **The same shapes as CUDA's**: `Device`, `Buf<S>`, `DMat<S>`, `DBatch<S>` carry the numbers `kernels/gpu/device`'s
  do, so what runs on `kernels/gpu` (ad/mirror, dyn/mirror, llm/gpu) ports kernel by kernel.

## What is there (slice 1)

`Gemm` and `GemmBatch` (every transpose, alpha and beta, heads and stacked matrices), the element maps (ReLU, GELU,
GELU's tanh form, SiLU, sigmoid, tanh, softplus, exp, affine), `Add` and `Mul` (with a row's bias repeated), `Softmax`
with the causal mask, `RmsNorm`, `LayerNorm`, `Gather` (an embedding's rows), `ArgMax` and `Patches` (a convolution's patch matrix, for a product to finish): every
kernel a transformer decoder's inference and Whisper's encoder and decoder need, F32. `make vktest` checks each against the CPU's kernels and runs a small decoder's forward
pass on the device against the CPU (logits within 1e-6 on lavapipe).

**LLM decoding** (`llm/vk.olang`, `oann-gen --device vulkan`): `llm/gpu`'s decoder on this backend - the weights and
the paged key-value cache on the device, rotary positions, QK-norm, grouped heads, biases, a tied or separate head -
for the dense decoders (Qwen3, Qwen2, Llama, Mistral, Granite, SmolLM3, Phi). Its ten kernels are its own, written with
`spirv.Kernel` in llm/vk.olang (llm/README.md, "On a Vulkan device"). BF16 without half floats: the weights and the
cache are BF16 held two to a 32-bit word, taken apart with a shift in the kernel (`Bits`, `FromBits`), so the memory is
half of F32's on every device, with no `shaderFloat16` or 16-bit storage asked for, and the logits are the CPU's to
BF16's own rounding. A matrix larger than the device binds as one buffer (`maxStorageBufferRange`: 128 MiB on software
Vulkan and some phones) is held in runs of its rows, a dispatch each. `make vktest` checks it on tiny models;
`make vkllm ARGS=MODEL_DIR` checks and times a real one.

Tested on software Vulkan (Mesa's llvmpipe) and, as Windows programs, on an NVIDIA RTX 4060 and an AMD Radeon
integrated GPU (2026-11-28: the decoder's logits within 4.8e-7 of the CPU's on both). Intel's and phones' GPUs have
not seen it.

## The product

`kern.Plan` says how a device computes `Gemm`: an invocation computes Rows x Cols elements of the result, a workgroup
is Side x Side invocations, and the operands are read **direct** (each step of the depth reads Rows elements of A and
Cols of B from the buffers) or **staged** (the workgroup first copies its tile of A's rows and B's columns, Chunk
steps of the depth at a time, into workgroup memory - `spirv`'s `Shared` arrays - meets at a `Barrier`, and every
invocation computes from there, so an element is read from the buffer once a workgroup). Every plan sums each element
in the depth's order: the result's bits do not depend on the plan. A plan may also be **fused** (`Fma`: a step's
product and sum in one operation), which a device rounds its own way; no default plan is.

`bench/vktune` (`make vktune`) times 41 plans on a device. GFLOPS F32, operands resident, 2026-11-28, one PC (Ryzen
5 7600: its two-unit Radeon integrated GPU, and an RTX 4060 with a training job on it meanwhile); the last two
columns are a Whisper encoder's linear layers (B a transposed weight matrix):

| plan | 256^3 | 1024^3 | 2048^3 | 1500 x 768 x 3072 | 1500 x 3072 x 768 |
|---|---|---|---|---|---|
| Radeon: 4x4 of 8x8, direct | 57 | 48 | 47 | 31 | 19 |
| Radeon: 8x8 of 8x8, direct | 119 | 99 | 97 | 41 | 25 |
| Radeon: 4x4 of 8x8, staged 16 | 139 | 148 | 148 | 128 | 128 |
| Radeon: **8x8 of 8x8, staged 16** | 164 | 194 | 196 | 188 | 188 |
| Radeon: 8x8 of 8x8, staged 16, fused | 216 | 261 | 229 | 224 | 222 |
| 4060: 4x4 of 8x8, direct | 339 | 1604 | 1615 | 592 | 600 |
| 4060: 4x4 of 16x16, staged 16 | 625 | 1960 | 2370 | 1756 | 2117 |
| 4060: **8x8 of 8x8, staged 16** | 368 | 1893 | 2458 | 1908 | 2028 |
| 4060: 8x8 of 8x8, staged 16, fused | 422 | 1991 | 2749 | 1911 | 2111 |

What it says. Staging is what counts: direct reads collapse where each invocation streams its own rows of a
transposed operand (a linear layer), and staged tiles take the Radeon from 19-47 to 188-196 and the 4060 from
600-1615 to 1900-2460. 8 x 8 blocks beat 4 x 4 (fewer reads a product); chunks of 8 and 16 steps tie and 32 loses;
16 x 16 workgroups gain only on small results on the 4060. Fusing gives 5-20% and is left off: the product then
keeps the CPU's arithmetic. cuBLAS on the same 4060 does 3283 and 3530 at 1024^3 and 2048^3 (`kernels/gpu`).

`Plan.For(d)` is the large products' plan: 8x8 of 8x8 staged 16 on a GPU, 8x8 of 4x4 direct on software Vulkan
(llvmpipe: 35 against 26 for 4 x 4 blocks at 512^3; a barrier there is a coroutine switch). `OANN_VK_GEMM=
ROWS,COLS,SIDE,CHUNK,FUSED` names another. A `Set` picks by the result's shape, since a staged workgroup computes
its whole tile: a result with a side under half a tile is read direct in 8 x 8 blocks, one under 8 rows or columns in
blocks of 1 along that side (a decoder step's one row), and on a discrete GPU a result under 512^2 elements takes 4x4
of 16x16 staged.

An integrated GPU against its CPU: the same Ryzen's CPU does about 100 GFLOPS a thread and 320-380 at 4 threads
(AVX-512), so this two-unit Radeon at 190 does not beat it; integrated GPUs of 8-12 units (laptops' Radeon 680M and
780M) have 4-6 times its arithmetic, and no such chip has been measured.

### A language model's product

`llm/vk`'s linear product (`LinPlan`: rows and outputs an invocation, the workgroup, the weights' words staged) was
measured the same way (`make vklintune`), the weights BF16 two to a word. GFLOPS, 2026-11-28, the same PC:

| rows x depth x outs | Radeon, direct | Radeon, best | 4060, direct | 4060, best |
|---|---|---|---|---|
| 1 x 1024 x 3072 | 0.7 | 13.8 (1x1 of 64x1, staged 16) | 35 | 75 (the same) |
| 1 x 1024 x 32768 | 0.7 | 15.3 (the same) | 50 | 109 (the same) |
| 8 x 1024 x 3072 | 29 | 71 (8x4 of 64x1, staged 16) | 250 (4x4 of 8x8, direct) | 250 |
| 32 x 1024 x 3072 | 64 | 169 (8x4 of 16x4, staged 16) | 807 (4x4 of 8x8, direct) | 807 |

A step of one row is bound by the work a weight costs (a word read, taken apart, two products), not by the device's
arithmetic: an invocation an output, the words staged so that a workgroup reads what lies together; each doubling
of the chunk from 4 to 16 words doubled the Radeon's speed, and longer chunks are not measured. `LinPlan.For` and
`ForMany`: one row and up to 3, 1x1 of 64x1 staged 16 on every GPU; 4 rows and more, 8x4 of 64x1 staged 16 on an
integrated GPU (8x4 of 16x4 from 16 rows) and 4x4 of 8x8 direct on a discrete one; software Vulkan reads direct.
`OANN_VK_LLM_ROW` and `OANN_VK_LLM_ROWS` ("ROWS,COLS,SX,SY,CHUNK") name others.

### Long submissions

A driver gives up on a submission that runs for seconds (Windows resets a device after two: `VK_ERROR_DEVICE_LOST`),
and an integrated GPU takes that long over a Whisper encoder recorded whole. `Device.Spend` counts the arithmetic the
products record and runs what is recorded once it passes `Device.Budget` (2e9 operations on an integrated GPU, 32e9
on a discrete one, no limit on software Vulkan), and every dispatch counts a share of it (a run of 8 kernels on an
integrated GPU, 512 on a discrete one), so a program need not `Sync` by the clock. `OANN_VK_TRACE=1` runs
every kernel as it is recorded and names the first that fails.

## Whisper on it

`speech/whispervk` runs speech/whisper's `Encode`, `Cross` and `Step` on a device: the weights (F32), every array of
the encoder and of a decoder's step, the decoder's cache of keys and values and the cross-attention's keys and values
are device buffers; a step sends its rows' tokens and positions up and brings its logits back, one `Sync` a step.
`speech/transcribe` chooses with `Whisper.LoadOn(dir, Backend.Vulkan)` (or `Backend.VulkanEncoder`: the encoder on the
device, the decoder on the CPU), `oann-listen --device vulkan`. Against the CPU on software Vulkan: a random model's
logits within 3e-6, whisper-tiny's within 2e-4, and the same transcript.

Speed on an integrated GPU (2026-11-28, a Ryzen 5 7600's two-unit Radeon, Windows; `whispervk`, random weights): the
small shape's encoder over 30 s takes 6.5 s and a greedy step 483 ms; medium's 19.5 s and 2.5 s. The same PC's CPU
takes about 1 s for small's encoder once built right (its matmul does 470 GFLOPS at 4 threads against the Radeon's
190; the times printed beside these were a Windows build that had lost its vector code: 7.7 s and 36 ms), so on that
machine the CPU is the faster place for both. The decoder's step is slow on the device because its one-row products
go through `Set.Gemm`'s direct plans, which an AMD GPU reads at under 1 GFLOPS; `llm/vk`'s staged one-row plan is
what they want (item 2 below). On software Vulkan (llvmpipe, 4 cores) the device is slower than the CPU's kernels,
as it must be (`make whispervk ARGS="small 3"`).

## What comes next

1. **The product on more devices**: the plans are measured on one Radeon and one NVIDIA card ("The product"); an
   Intel integrated GPU and a larger Radeon want `make vktune`'s table before their plan is trusted. Fused steps and
   `Plan.For` by vendor follow from those.
2. **Whisper, tuned on real GPUs** (first: transcription on an office laptop's integrated GPU): it runs (above);
   what it wants next is a staged plan for its decoder's one-row products (as `llm/vk`'s) and its weights packed as
   item 4 says.
3. **LLM decoding, tuned on real GPUs**: the decoder runs (above); its product (`llm_row`, `llm_rows`) wants
   plans measured as the product's were, the norm and the softmax are one invocation a row and want a parallel sum, and sampling could
   stay on the device (the logits come down every step). `device.Backend.Vulkan` in `kernels/device` once plans run
   there.
4. **Packed BF16 elsewhere**: llm/vk's two-to-a-word BF16 (no device feature needed) for Whisper's weights and the
   general product; F16 where a device has `shaderFloat16` and 16-bit storage only if it measures faster.
5. **Training**: ad/mirror's backward kernels and AdamW, kernel by kernel.
6. **Windows and Android builds** of the shim (the loader's name is already chosen per target), and a Metal backend
   for Apple.
