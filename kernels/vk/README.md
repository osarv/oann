# kernels/vk

oann's kernels on any GPU with a Vulkan driver: AMD, Intel, NVIDIA, integrated GPUs, phones', and a CPU's software
Vulkan (Mesa's lavapipe). The design is `docs/vulkan.md`. `kernels/gpu` is the CUDA backend, for NVIDIA alone.

Lightly tested: every test passes on software Vulkan; real hardware has seen little of it. Expect bugs on a driver it
has not met.

| file | what |
|---|---|
| `vk.c` | the C shim over Vulkan, the one place its structs are filled: a device, buffers, kernels (compute pipelines) and the queue, everything crossing as numbers. It opens the Vulkan loader when a device is asked for, so a build needs Vulkan's headers only and the program runs where there is no Vulkan |
| `device.olang` | `Device` (`Open`, `Sync`, `Close`; `Devices()` lists them), `Buf<S>` (`Make`, `Upload`, `Download`, `Zero`, `CopyFrom`), the views `DMat<S>` and `DBatch<S>` (kernels/mat's `Mat` and kernels/gemm's `Batch` over device memory), `VkError` |
| `spirv.olang` | `Kernel`: a compute shader written in olang - buffers, parameters, arithmetic, a float's bits (`Bits`, `FromBits`), variables, `If` and `Loop` - and assembled to SPIR-V 1.0 by `Finish`. No shader compiler and no shader text |
| `kern.olang` | `Kernel.Make` (a pipeline of a written kernel) and `Set`, a device's kernels: `Gemm`, `GemmBatch`, `Map` (ReLU, GELU, GELU's tanh form, SiLU, sigmoid, tanh, softplus, exp), `Affine`, `Add`, `Mul`, `Softmax` (with the causal mask), `RmsNorm`, `LayerNorm`, `Gather`, `ArgMax`, `Patches` (a convolution as a product). F32 |
| `lm_test.olang` | a transformer decoder's forward pass on the device against the CPU |

## Use

```
import "github.com/osarv/oann/kernels/vk/device"
import "github.com/osarv/oann/kernels/vk/kern"

d := try device.Device.Open()                 # the best device here; VkError.NoDevice where there is none
ks := try kern.Set.Make(d)                    # its kernels, compiled by the driver now
a := try device.Buf<F32>.Make(d, m * k)
a.UploadAll(hostA)
...
ks.Gemm(d, device.DMat<F32>(c, 0, m, n, n), device.DMat<F32>(a, 0, m, k, k), false, device.DMat<F32>(b, 0, k, n, n), false)
c.DownloadAll(hostC)
try d.Sync()                                  # everything has run; hostC is read after this
```

Build with `-define Vulkan=true` (Vulkan's headers: `libvulkan-dev` on Debian and Ubuntu). `Device.Open()` takes the
first discrete GPU, else an integrated one, else software Vulkan; `OANN_VK_DEVICE=N`, or `Device.Open(N)`, picks one of
`Devices()`. Everything asked of a device is recorded in program order and runs by the next `Sync`.

`llm/vk.olang` is a language model's decoder on these (`oann-gen --device vulkan`); its kernels are written there.

`make vktest` runs the tests, `make vkgemm` times the product against the CPU, `make vkllm` checks and times a
language model's decoding. `OANN_VK_DUMP=DIR` writes each kernel's
SPIR-V to `DIR/NAME.spv` (for `spirv-val` and `spirv-dis`).
