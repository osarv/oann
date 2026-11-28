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

Tested on software Vulkan (Mesa's llvmpipe). Real GPUs have seen little of it.

## What comes next

1. **A faster product.** An invocation computes a 4 x 4 block of the result, reading four elements of each operand a
   step of the depth for sixteen products (5x the one-element form on llvmpipe; `make vkgemm`). Next, measured on real
   GPUs first: tiles of the operands staged in workgroup memory, and the block's size by device.
2. **Whisper on it** (first: transcription on an office laptop's integrated GPU): speech/whisper's `Encode` and `Step`
   with their arrays on the device - `Patches` and `Gemm` for the two convolutions, `Map` GELU, `LayerNorm`, `GemmBatch`
   and `Softmax` for the attention, the decoder's key and value cache as device buffers - behind a device choice in
   speech/transcribe, checked against the CPU's logits.
3. **LLM decoding on it**: `llm/gpu`'s decoder (the paged cache, rotary positions, QK-norm) on `kernels/vk`, and
   `device.Backend.Vulkan` in `kernels/device` once plans run there.
4. **Half precision** where the device has `shaderFloat16` and 16-bit storage (BF16 has no Vulkan form: F16).
5. **Training**: ad/mirror's backward kernels and AdamW, kernel by kernel.
6. **Windows and Android builds** of the shim (the loader's name is already chosen per target), and a Metal backend
   for Apple.
