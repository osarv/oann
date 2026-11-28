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

Tested on software Vulkan (Mesa's llvmpipe). Real GPUs have seen little of it.

## Whisper on it

`speech/whispervk` runs speech/whisper's `Encode`, `Cross` and `Step` on a device: the weights (F32), every array of
the encoder and of a decoder's step, the decoder's cache of keys and values and the cross-attention's keys and values
are device buffers; a step sends its rows' tokens and positions up and brings its logits back, one `Sync` a step.
`speech/transcribe` chooses with `Whisper.LoadOn(dir, Backend.Vulkan)` (or `Backend.VulkanEncoder`: the encoder on the
device, the decoder on the CPU), `oann-listen --device vulkan`. Against the CPU on software Vulkan: a random model's
logits within 3e-6, whisper-tiny's within 2e-4, and the same transcript.

Speed is not known on a real GPU. On software Vulkan (llvmpipe, 4 cores) it is slower than the CPU's kernels, as it
must be: the small shape's encoder over 30 s takes 15.0 s against 1.6 s, a greedy step 122 ms against 44 ms
(`make whispervk ARGS="small 3"`). Whether an integrated GPU beats the CPU is for that bench to say on one; the
product (item 1 below) is the encoder's whole cost, and a step is about 25 small dispatches a layer, so a device's
cost a dispatch decides the decoder.

## What comes next

1. **A faster product.** An invocation computes a 4 x 4 block of the result, reading four elements of each operand a
   step of the depth for sixteen products (5x the one-element form on llvmpipe; `make vkgemm`). Next, measured on real
   GPUs first: tiles of the operands staged in workgroup memory, and the block's size by device.
2. **Whisper, tuned on real GPUs** (first: transcription on an office laptop's integrated GPU): it runs (above);
   what it wants next is the product's tiles of item 1 and its weights packed as item 4 says.
3. **LLM decoding, tuned on real GPUs**: the decoder runs (above); its product (`llm_row`, `llm_rows`) wants the
   tiles item 1 finds, the norm and the softmax are one invocation a row and want a parallel sum, and sampling could
   stay on the device (the logits come down every step). `device.Backend.Vulkan` in `kernels/device` once plans run
   there.
4. **Packed BF16 elsewhere**: llm/vk's two-to-a-word BF16 (no device feature needed) for Whisper's weights and the
   general product; F16 where a device has `shaderFloat16` and 16-bit storage only if it measures faster.
5. **Training**: ad/mirror's backward kernels and AdamW, kernel by kernel.
6. **Windows and Android builds** of the shim (the loader's name is already chosen per target), and a Metal backend
   for Apple.
