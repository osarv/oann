# llm

Inference for openly licensed decoder-only language models on the CPU: Hugging Face checkpoints of the Qwen2 and Qwen3
architectures loaded as they are published, run in batches on AMX tiles, and `oann-gen`, a command that writes
completions of a file of prompts as one long, resumable job. Nothing here trains; the weights are never committed.

| file | what |
|---|---|
| `model.olang` | `Config` (config.json), `Model.Load` (safetensors, one file or the shards of an index; BF16, F16 or F32 stored), `Step` (any rows of any sequences through every layer), `Cache` (the paged key-value cache), `State` (a step's rows and buffers) |
| `bpe.olang` | `Bpe`, Hugging Face's byte-level BPE (tokenizer.json): the added tokens, the Qwen2 pre-tokenizer pattern over Unicode classes (`unicode.olang`), the merges by rank |
| `engine.olang` | `Engine`, continuous batching: requests admitted as slots and pages allow, prompts read in chunks beside the sequences being decoded, preemption when the pages run out; `Sample` (temperature, top-k, top-p) |
| `chat.olang` | `Chat`: the chat template (ChatML, DeepSeek-R1's), stop tokens and sampling defaults of a model directory; `Fetch` from Hugging Face |
| `gen.olang` | `oann-gen` |

## Using it

```
make oanngen                                   # builds build/oann-gen
build/oann-gen Qwen/Qwen3-4B prompts.jsonl out.jsonl --n 4 --temp 0.6 --max-tokens 8192
```

`MODEL` is a directory holding a checkpoint or a repository's name, fetched once over HTTPS
(`https://huggingface.co/<repo>/resolve/main/<file>`, by curl) into `$OANN_MODELS` (default `~/.cache/oann/models`,
the repository's `/` written `--`). Each line of `prompts.jsonl` is an object with `"prompt"` (one user turn, put through
the model's chat template, with `"system"` or `--system` before it) or `"text"` (read as it is), and an optional `"id"`
(the line's number otherwise). Each completion is a line of `out.jsonl`, written as soon as it finishes:

```
{"id": "q17", "sample": 2, "text": "<think>\n...\n</think>\n\n...", "prompt_tokens": 41, "tokens": 1873, "finish": "stop"}
```

`finish` is `stop` (a stop token: generation_config's `eos_token_id` and the end of a turn) or `length`. Thinking
models write their reasoning first, between `<think>` and `</think>`; DeepSeek-R1's template opens the `<think>` itself,
so it is put at the front of `text` and both families read alike. `--no-think` turns Qwen3's thinking off.

The job resumes: run again with the same arguments, it skips the (id, sample) pairs already in `out.jsonl` and makes
the rest. Each completion draws from its own generator, seeded by `--seed`, its id and its sample number, and a row of
a step never reads another row, so a completion does not depend on the batch it ran in or on an interruption.

| flag | |
|---|---|
| `--n K` | completions of each prompt (1) |
| `--temp T`, `--top-p P`, `--top-k K` | sampling; the model's generation_config.json by default (0.6, 0.95, 20 for Qwen3); `--temp 0` is greedy |
| `--max-tokens M` | tokens a completion may have (4096) |
| `--batch B` | sequences decoded at once (64) |
| `--ctx C` | positions a sequence may have, prompt and completion (default: the longest prompt and M) |
| `--kv-gb G` | memory for the key-value cache (default 90% of what is free after loading, less 1 GB) |
| `--int8` | 8-bit weights, a scale per 32 inputs (the head stays BF16): 0.56 of the memory, needs AMX (BF16 without it); the 8B models need it on 16 GB |
| `--exact` | F32 activations and cache: slow, for checking against a reference |

## How it works

A **step** takes any set of rows - each a token of some sequence at its position - and runs them through the model
together: every product of the step is shared by all its rows, so the weights are read from memory once a step whatever
the batch. Decoding is bound by that read (the weights of a 1.5B model are 3.4 GB), so a step of 64 sequences costs
little more than a step of one - until the contexts grow, when reading the key-value cache takes over. Prompts are read in the same steps, up to 256 rows a step beside the sequences being
decoded, so a new sequence starts the moment another finishes (continuous batching).

**Products.** On a CPU with AMX (`TargetHasTileDot`), the weights stay BF16 and are packed once, as they are loaded,
into the pair layout of std/tile's `AddDot`, in blocks of 32 outputs: a step rounds its activations to BF16 and each
task adds `X W^T` into its blocks of the result, summing in F32. With `--int8` the layers' weights are 8 bits with a
scale per 32 inputs (symmetric), widened to BF16 512 inputs of a block at a time into the L1 cache as they are used. Without AMX, or with `--exact`, the activations stay
F32 and kernels/gemm multiplies them by the weights widened.

**Attention** reads a paged key-value cache: pages of 64 positions, BF16, taken from a pool as a sequence grows and
given back when it ends. A page's keys for a head are kept in AddDot's pair layout over the head's dimensions and its
values over the positions, so a row's scores against a page (its head group's queries, padded to 16 rows, times the
page's keys) and the page's share of the output (the weights in BF16 times the values) are one AddDot each, the softmax
between them in F32. When the pool runs dry, the youngest sequence is preempted - its pages freed, its request queued
again from its prompt, to draw the same tokens.

**The architectures.** RMSNorm; rotary positions with the model's theta (and linear rope scaling); grouped-query
attention; a SwiGLU perceptron; Qwen2's biases on the query, key and value projections; Qwen3's RMSNorm of each query
and key head before the rotation; the head tied to the embedding or its own. YaRN and the mixture-of-experts models are
not run.

**Tokens.** tokenizer.json's byte-level BPE: added tokens (`<|im_start|>`, `<think>`, ...) are matched in the text
first, the stretches between them split by the pre-tokenizer's pattern (Qwen2's, one digit a chunk) over the Unicode
letter and number classes, and each chunk's bytes merged lowest rank first. It encodes as the tokenizers library does
(checked on text in English, German, Chinese, Japanese, Korean, Russian, code, emoji and odd whitespace); the NFC
normalizer is not applied, which matters only for text not already in NFC.

## Checks

The tests (`make test`) build small random Qwen2 and Qwen3 checkpoints and hold the AMX path to the exact one within
BF16's rounding, decoding through the cache a token a step to reading the prompt at once (over pages), a row's logits
to be the same, bit for bit, alone or beside other sequences, a batch with prompts in chunks and with preemption to
generate what each request does alone, stop tokens, and the sampler's distribution; and pin that a step allocates
nothing. Where `$OANN_MODELS` holds `Qwen--Qwen3-0.6B` and `deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B`, a test holds
the exact path to Hugging Face transformers (F32) on a fixed prompt - the last position's logits within 1e-3 and ten
greedy tokens the same - and the AMX path to the same tokens. On the whole vocabulary the exact path's logits are
within 1e-4 of transformers' for both models; the AMX path's (BF16 activations and cache) within 0.12 for Qwen3-0.6B
and 0.36 for the distilled 1.5B, whose keys carry large biases - the same greedy tokens either way.

## Speed

Decoding on a 4-core cloud VM with AMX (Xeon, 15 GB, about 25 GB/s from memory in practice), tokens a second for every
sequence of the batch at a context of 512 positions (`make llmbench ARGS="DIR 512 16 bf16 1 16 64"`, bench/llm.olang),
and the process's peak memory at batch 64:

| model | weights | batch 1 | batch 16 | batch 64 | peak |
|---|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | BF16, 3.4 GB | 5.3 | 86 | 228 | 4.0 GB |
| DeepSeek-R1-Distill-Qwen-1.5B | INT8, 2.3 GB | 5.0 | 73 | 197 | 2.9 GB |
| DeepSeek-R1-Distill-Qwen-1.5B, context 2048 | BF16 | | 89 | 183 | 5.4 GB |
| Qwen3-4B | BF16, 7.7 GB | 3.2 | 40 | 89 | 10.7 GB |
| Qwen3-4B | INT8, 4.6 GB | 1.9 | 31 | 71 | 7.7 GB |
| Qwen3-8B | INT8, 9.8 GB | 1.1 | 14 | 43 | 12.9 GB |

A step's time goes mostly to the products, which read the weights from memory: a batch of 64 costs about 1.3 to 1.6
times a batch of one. The rest is attention, which reads the cache and grows with the context - at 2048 positions and
batch 64 it is as long as the products for the 1.5B model, sooner for models with more key-value heads (Qwen3's 8). INT8
saves memory, not time here: widening the weights costs what reading half the bytes saves. Qwen3-8B in BF16 (16 GB)
does not fit in 15 GB. `oann-gen` on 64 prompts with the 1.5B model, batch 64, at most 1024 tokens: 60194 tokens in
301 s, 200 tokens a second, peak 5.2 GB. Measurements on a shared VM vary by 10-20% from run to run.

A prompt is read in the same steps, each of its positions a row with attention of its own; that suits prompts of
hundreds of tokens, not of many thousands.

## Models and licences

The weights are fetched from Hugging Face by whoever runs the command; none are in this repository. Models of these
architectures known to run, and their licences (each model's own LICENSE file is fetched beside it):

| model | architecture | licence |
|---|---|---|
| Qwen/Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B | qwen3 | Apache 2.0 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B | qwen2 | MIT (built from Qwen2.5, Apache 2.0) |
| Qwen/Qwen2.5-0.5B-Instruct ... 7B-Instruct | qwen2 | Apache 2.0 |

Check a model's licence before using what it generates.
