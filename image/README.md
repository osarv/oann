# image

Text to image with FLUX.2 klein (Black Forest Labs' distilled rectified-flow models; the 4B model's weights are
Apache 2.0), run from the model's Hugging Face repository as it is published. The transformer runs on the CPU or on any GPU with a
Vulkan driver; the text encoder and the decoder on the CPU. The weights are never committed.

| file | what |
|---|---|
| `klein.olang` | the pipeline: `Embed` (the prompt through the model's Qwen3 text encoder, llm/), `Flow` (noise to latents in a few Euler steps), `Decode` (latents to pixels), `Generate` (all three), `Sigmas` (the steps' noise levels); `Transformer` and `Steps`, `Decoder` and `Pixels`, the same with the model loaded once by the caller |
| `flux2.olang` | `Model`, the flow transformer (diffusers' Flux2Transformer2DModel): `Load`, `Run` (the buffers of one image size), `SetText`, `Forward` (one pass: every latent token's velocity) |
| `vk.olang` | `Gpu`, the transformer on a Vulkan device (kernels/vk): `Make` (the buffers, the blocks that fit kept on the device), `SetText`, `Forward`; built with `-define Vulkan=true` |
| `vae.olang` | `Vae`, the autoencoder's decoder (AutoencoderKLFlux2): `Load`, `Unpatch` (tokens to latents), `Decode` |
| `png.olang` | `Encode`: pixels as a PNG file |
| `gen.olang` | `oann-image`, the command, and its server mode |
| `tools/ref.py`, `tools/check.olang` | the reference (diffusers, F32, CPU) and the program that holds each stage to it |

## Using it

```
make oannimage                                 # builds build/oann-image
hf download black-forest-labs/FLUX.2-klein-4B --local-dir klein4b --exclude flux-2-klein-4b.safetensors "*.jpg"
build/oann-image klein4b "A cat holding a sign that says hello world" cat.png --size 512x512
```

`MODEL` is the directory: `tokenizer/`, `text_encoder/`, `transformer/`, `vae/` (15.9 GB for klein 4B: the text encoder
8.0, the transformer 7.8, the decoder 0.2). `--size WxH` (1024x1024 by default, each side a multiple of 16), `--steps N`
(4, what the distilled models are made for), `--seed S` (the noise; a seed gives the same picture again), `--exact`
(F32 activations). The noise is std/rand's, so a seed's picture is not the picture diffusers draws for that seed.

`--device vulkan` runs the transformer on a GPU (a build with `-define Vulkan=true`, which `make oannimage` adds where
Vulkan's headers are; `OANN_VK_DEVICE=N` picks one of several), with F32 activations as `--exact`.

From a program: `klein.Generate(dir, prompt, width, height)` gives the picture's bytes (row-major, red, green, blue).

**Server mode** keeps the transformer and the decoder loaded for many pictures, one JSON object a line on stdin:

```
build/oann-image klein4b --serve --device vulkan
{"prompt": "A cat holding a sign that says hello world", "out": "cat.png", "width": 512, "height": 512}
```

It answers each on stdout, one JSON object a line: `{"step":1,"of":4}` as each step starts, then
`{"ok":true,"path":"cat.png","width":512,"height":512,"seconds":41.2}`, or `{"ok":false,"error":"..."}` for a request
it cannot do. `width`, `height` (1024), `steps` (4) and `seed` (0) are optional. It ends when stdin closes. The text
encoder is loaded for each picture and let go, so the server holds the transformer's 7.8 GB and not the encoder's 6 GB
beside it.

**For Windows**: `olang build -define Vulkan=true -target x86_64-windows:x86-64-v3 -output build/oann-image.exe
image/gen.olang`, with mingw-w64 and Vulkan's headers in its include directory (`cp -r /usr/include/vulkan
/usr/include/vk_video /usr/x86_64-w64-mingw32/include/`). The program opens Windows' own Vulkan loader
(vulkan-1.dll, installed with the graphics driver).

## How it works

**The prompt** is one user turn of the text encoder's chat template with thinking off, padded to 512 tokens. The
encoder is a Qwen3 language model (4B for klein 4B) run by llm/, its first 27 layers only: a token's embedding is the
stream after layers 9, 18 and 27 side by side (`State.Keep`). Padding tokens attend to the prompt alone.

**The latents** are tokens of 128 numbers, one per 16 x 16 pixels. They start as noise; the transformer reads them with
the prompt's embeddings and the noise level and gives each token's velocity, and an Euler step moves the tokens along
it to the next level: four steps from 1 to 0 for the distilled models, the levels shifted toward noise for larger images
as the reference shifts them. The transformer's rows are the prompt's 512 tokens and the image's: five blocks with
weights of their own for text and image that meet in the attention, then twenty that run all rows as one stream
(flux2.olang's header has the details). Its weights stay BF16 as stored (7.8 GB); the activations are F32 with
`--exact` and BF16 otherwise, on AMX tiles where the CPU has them.

**The picture**: the tokens, un-normalized by the autoencoder's batch-norm statistics and split into 2 x 2 latent
positions, go through the decoder's convolutions (kernels/conv) to 8 x 8 pixels a position.

Each part loads its model and lets it go before the next, so the peak memory is the transformer's and its buffers, not
the sum.

**On a GPU** (vk.olang) the stream and every product, norm, rotation and attention are on the device; the host works
out the noise level's modulations (one row through a few small maps) and sends them up each step with the latents.
The 7.4 GB of the blocks' weights is more than most cards hold beside a desktop, so the first blocks that fit a budget
stay on the device and the rest are sent up again on each pass, in order with the work that reads them. The budget is
`OANN_IMAGE_VK_MIB` (MiB of device memory for everything), else half the device's memory. Attention is the device's
general product: a group of heads' scores in one buffer of at most 512 MiB, their softmax, the values' mean.

## Checks

`tools/ref.py` runs diffusers' Flux2KleinPipeline stage by stage on the CPU in F32 and writes what each stage read and
wrote; `tools/check.olang` runs oann's stages on the same inputs:

```
python image/tools/ref.py klein4b ref/cat256 text && python image/tools/ref.py klein4b ref/cat256 flow && python image/tools/ref.py klein4b ref/cat256 vae
olang build -output build/image_check image/tools/check.olang
build/image_check klein4b ref/cat256 text exact
build/image_check klein4b ref/cat256 flow exact
build/image_check klein4b ref/cat256 vae
build/image_check klein4b ref/cat256 all exact out.png
```

Measured 2026-11-28 on a 4-core Xeon VM with AMX (15 GB), klein 4B, "A cat holding a sign that says hello world",
256 x 256, 4 steps, the reference's noise:

| stage | difference from the reference |
|---|---|
| tokens | none (512 of 512) |
| prompt embeddings, exact | mean 2.6e-5, worst 0.03, of values up to 16392 |
| velocity, exact, steps 1-4 | worst 3e-5, 1.4e-4, 2.8e-4, 4.1e-4 of values up to 8 |
| latents after 4 steps, exact | worst 3.2e-4 of values up to 5.2 |
| decoder (the reference's latents) | worst 1.9e-6 of values up to 1.17 |
| the picture, prompt to pixels, exact | mean 0.014 of 255 a channel, worst 20 |
| the picture, BF16 activations | mean 4.2 of 255: the same picture to the eye, details differ (rounding grows over the steps, as it does for the reference run in BF16) |

Times there, 256 x 256 (768 rows a pass): exact - text 34 s, flow 104 s (42 s of it loading; 14.5 s a step), decoder
3.7 s; BF16 - text 17 s, flow 71 s (30 s loading; 9.5 s a step), decoder 3.6 s. Larger sizes and a GPU are not
measured yet.
