# models

The reference models of oann (DESIGN.md): each a config, the layers made from it (`New`
registers them on an `ad.Graph` before `Plan`), a `Logits`/`Task`/`Lm` that records the forward pass and hands the
loops (`nn/train`, `nn/generate`, `nn/quant`) their handles. They are built from `nn/layers` and tested in F32 and BF16
from their first commit; the numbers they train to are in the table below, and the programs that measure them are in
`examples/`.

| file | what | trains to |
|---|---|---|
| `mlp.olang` | `MlpConfig`, `Mlp`: dense layers with an activation between them, softmax cross-entropy; `MlpConfig.Mnist()` is 784-128-10 (101 770 parameters); `Int8` makes its `quant.Model` | MNIST 97.73% after 10 epochs, F32 and BF16 (`examples/mnist_mlp.olang`) |
| `cnn.olang` | `CnnConfig`, `Cnn`: stages of a 3x3 convolution (padding 1), ReLU and 2x2 max pooling, then a dense layer; `CnnConfig.Mnist()` is 16 and 32 channels (20 490 parameters); `Int8` | MNIST 97.29, 97.93, 98.48% after epochs 1 to 3 (`examples/mnist_cnn.olang`) |
| `gpt.olang` | `GptConfig`, `Gpt`: GPT-2's decoder with options - rotary positions, RMS norms, a gated SwiGLU perceptron, an untied head, dropout; `GptConfig.Char` is the character transformer, `GptConfig.Yardstick` the 10.4 million parameter model; `Config.Params()` is the parameter count | tiny Shakespeare validation loss 1.7961 nats a character after 2000 steps (`examples/charlm.olang`); TinyStories 2.880 nats a token at 1.47 million tokens (`examples/tinystories.olang`) |
| `ssm.olang` | `SsmConfig`, `Ssm`: a token embedding, blocks of Mamba's selective scan (`layers.SsmBlock`) or the gated delta rule (`layers.DeltaBlock`), a final norm, the tied head; no positions to learn; `SsmConfig.Yardstick` is 9 blocks of 384 (10.2 million parameters) | tiny Shakespeare 1.5725 after 2000 steps (`examples/charlm_ssm.olang`); TinyStories 2.760 at 1.47 million tokens (`examples/tinystories.olang`) |
| `qwen.olang` | `Qwen`: Qwen3 (and Qwen2, with its query, key and value biases) as llm/model loads it - embeddings, RMS norms, each query and key head RMS-normed (Qwen3), rotary positions at the config's theta, grouped-query attention (each key and value head repeated by an Expand view), SwiGLU, a tied or separate head - every weight a frozen one (`g.Frozen`), filled from `llm/model`'s loaded checkpoint | not trained itself: the base of fine-tuning, by LoRA adapters (`AddLora`) on any of its seven projections; its logits are llm/model's to 4e-7 of the largest in F32 (the oracle test); LoRA on Qwen3-0.6B (rank 8, every projection, 5.0 million parameters) takes one 16-token sentence from loss 2.66 to 0.003 in 8 AdamW steps at 1e-3, ~1.5 s a step on 4 CPU cores (`examples/qwen_lora.olang`) |

A model's parameters are registered in one order - embedding, positions, blocks, final norm, head - so a checkpoint
(`g.Save`) names them `param.K` by that order and loads into any graph of the same model; `Gpt` is exactly
`train.LanguageModel.Gpt`'s model (a test holds the two to the same parameters and logits). A model with a scan block
generates by `generate.Rerun` (a recurrence keeps no state between `Forward` calls); a GPT generates through the key-value
cache (`graph.Graph<E>(1, 1, Mode.Captured, true)` with `Lm(g, false)`).

Each file's tests hold its parameter count to the graph's, a gradient check where the model is new (the CNN and the SSM
through the whole network), a short training run on a task a small model solves (the loss falls, a held-out set is classified
right, the run repeats bit for bit), the key-value cache against the whole forward (GPT, rotary and gated included) and
a step that allocates nothing (the resident memory does not grow over 20 steps after the first). `olang test
models/FILE.olang`; `make test` runs them all.

`Qwen` is built from a checkpoint rather than trained: `Qwen.New(g, cfg, ctx)?` records it (cfg is `llm/model`'s
`Config`; scaled rotary positions are refused), `q.Lm(g)` gives the tokens, logits and loss, and after `Plan`
`q.Fill(g, m)` copies the weights of `model.Model.Load(dir, true)` into the graph - in the graph's element type, so a
`Graph<BF16>` holds them in BF16. Its weights are frozen: no gradients or optimizer state of their own, not among
`Params()`, uploaded to the GPU once; gradients flow through them to whatever is trained beside them. Its test runs a
random Qwen3-shaped checkpoint (tied and untied heads, fewer key-value heads than query heads, Qwen2's biases) through
both `llm/model` and the graph, F32 and BF16; ad/gpu_test runs it on the GPU against the CPU.

For QLoRA (nn/qlora): `Qwen.New(g, cfg, ctx, quantized, recompute)` makes the seven projections 4-bit NF4 weights
(`g.NF4`, quantized from F32 as filled) and recomputes each block in the backward (`g.Recompute`); `q.Lm(g, true, chunk)`
gives the loss by `g.ChunkedCE` over the final norm's output and the head (the embedding where tied), no logits.
`q.FillFrom(g, dir)` reads the checkpoint's files (model.safetensors, or the shards model.safetensors.index.json names)
straight into the graph a tensor at a time - Fill's numbers bit for bit, without llm/model's copy - so a model fills
within its largest tensor in F32 beyond the graph. `Footprint.Of<T>(cfg, ctx, batch, quantized, recompute, chunk, lora,
backend)` lays such a graph out without allocating and gives the arena, the NF4 weights and AdamW's moments in bytes
(the arena equals what Plan allocates, tested). At context 1024, batch 1, rank 16 on every projection, NF4 and
recompute, chunk 8192, it plans on the GPU (BF16) Qwen3-0.6B 921 MB, 1.7B 2008 MB, 4B 3829 MB (CUDA's own context and
cuBLAS's workspace come on top).

It is fine-tuned by LoRA (`nn/lora`): `s := q.AddLora(g, lora.Config(8, 16.0, ["q_proj", "v_proj"]))?`, before `Lm`,
gives each projection named (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, as PEFT's target_modules
name them) an adapter, y = W x + (alpha / r) B A x with A [r, in] uniform in +-1/sqrt(in) and B [out, r] zero - so the
model starts as its base exactly - and A and B are the graph's only parameters, trained by the usual optimizers.
`s.Save(g, dir)` writes adapter_model.safetensors under PEFT's names
(`base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`) and adapter_config.json, which PEFT reads;
`s.Load(g, dir)` reads one back into a graph built with `lora.Config.Read(dir)`; `lora.Merge(base, dir, out)` writes a
copy of the base model with W += (alpha / r) B A in each weight's own dtype (`wide`: F32), which `llm/model` and oann-gen
run like any checkpoint. Its tests: at the start the adapted logits equal the base's bit for bit; only A and B are
parameters and their gradients match central differences in F64; a step leaves the frozen weights as they were and
20 steps lower the loss of a fixed batch; saved adapters load back bit for bit; the base graph given the merged F32
weights reproduces the adapted logits to 1e-5 of the largest, and `llm/model` on the merged model (holding its weights
in BF16) to 1%; ad/gpu_test trains LoRA on the GPU against the CPU step by step.
