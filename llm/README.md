# llm

Inference for openly licensed decoder-only language models on the CPU: Hugging Face checkpoints of the Qwen2, Qwen3,
Llama (Llama 3, SmolLM2), Mistral, Granite, SmolLM3 and Gemma 3 architectures and the mixtures of experts of Qwen3-MoE,
Granite-MoE and OLMoE loaded as they are published, run in batches on AMX tiles, and `oann-gen`, a command that writes
completions of a file of prompts as one long, resumable job. Nothing here trains; the weights are never committed.

| file | what |
|---|---|
| `model.olang` | `Config` (config.json), `Model.Load` (safetensors, one file or the shards of an index; BF16, F16 or F32 stored), `Step` (any rows of any sequences through every layer), `Cache` (the paged key-value cache), `State` (a step's rows and buffers) |
| `gguf.olang` | GGUF files (llama.cpp's): the header and metadata, rows of a tensor widened to F32 (F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K), the metadata as the config.json and tokenizer.json the rest reads; `Model.Load` and `oann-gen` take a `.gguf` path in place of a directory (llama, qwen2, qwen3, qwen3moe, granite, granitemoe, olmoe, smollm3; not Gemma) |
| `bpe.olang` | `Bpe`, Hugging Face's BPE (tokenizer.json), byte-level or SentencePiece's (Gemma: "▁" for spaces, byte fallback): the added tokens, the pre-tokenizer's pattern (Qwen2's, Llama 3's or GPT-2's, digits split first or not) over Unicode classes (`unicode.olang`), the merges by rank, `ignore_merges` |
| `engine.olang` | `Engine`, continuous batching: requests admitted as slots and pages allow, prompts read in chunks beside the sequences being decoded, preemption when the pages run out; `Sample` (temperature, top-k, top-p) |
| `chat.olang` | `Chat`: the chat template (ChatML, DeepSeek-R1's, Llama 3's, Granite's, Gemma's), stop tokens and sampling defaults of a model directory; `Fetch` from Hugging Face |
| `gen.olang` | `oann-gen` |
| `tools/reference.py`, `tools/tokref.py`, `tools/ggufref.py` | the references the tests are held to: transformers and tokenizers, F32, CPU (below, "Checks") |

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
| `--kv8` | values in the cache in 8 bits, a scale per position and head: 0.76 of the cache's memory, needs AMX |
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

**The architectures.** RMSNorm; rotary positions with the model's theta, scaled as `rope_scaling` (or transformers 5's
`rope_parameters`) says - linear, Llama 3.1's (long wavelengths divided by the factor, short ones kept, a blend between)
or YaRN's (frequencies interpolated by a ramp between beta_slow and beta_fast turns over the original context, cos and
sin times the attention factor); grouped-query attention, over a sliding window where the config sets one (Mistral's
`sliding_window`, Qwen2's with `use_sliding_window`, `layer_types`' "sliding_attention"); a SwiGLU perceptron; Qwen2's
biases on the query, key and value projections; Qwen3's RMSNorm of each query and key head before the rotation;
Granite's multipliers (the embedding, each residual branch, the scores in place of 1/sqrt(d), the logits divided);
SmolLM3's layers without rotation (`no_rope_layers`); the head tied to the embedding or its own, its rows padded to a
multiple of 16 (Granite's vocabulary of 49159); Gemma 3's text models (`gemma3_text`: 270M, 1B): the embedding times
sqrt(Dim), norm weights read as 1 + w, each branch's output normed before it is added, a GELU (tanh form) gate,
`query_pre_attn_scalar`, its sliding-window layers rotated by `rope_local_base_freq` and unscaled (`sliding_window_pattern`
or `layer_types`). Gemma 3's larger, multimodal checkpoints (4B and up: `language_model.` names, a `text_config`) and
Phi are not run yet.

**Mixtures of experts** (Qwen3-MoE, Granite-MoE, OLMoE). Each row's router logits (F32) go through a softmax; the top
k experts are kept, their weights renormalized where the config's `norm_topk_prob` says (Granite-MoE's softmax over the
k is the same thing). A step's (row, choice) pairs are grouped by expert, so an expert's weights are read once a step for
all the rows that chose it; on AMX whole experts are spread over the tasks, most rows first, each run on one core (an
expert's products are too small to split), and the outputs summed into the rows, weighted. Qwen3-MoE's dense layers
(`mlp_only_layers`, `decoder_sparse_step`) run the dense perceptron; OLMoE's QK-norm is over a row's whole queries and
keys. Shared experts (Qwen2-MoE) are not run. Because only k experts' weights are read per row, a mixture decodes much
faster than a dense model of its size:

| model | weights | batch 1 | batch 16 | batch 64 | peak |
|---|---|---|---|---|---|
| granite-3.1-3b-a800m-instruct (3.3B, 0.8B active) | BF16, 6.3 GB | 9.0 | 70 | 141 | 8.8 GB |
| granite-3.1-3b-a800m-instruct | INT8 | 7.1 | 50 | 110 | 6.1 GB |
| granite-3.3-2b-instruct (dense 2.5B), for comparison | BF16, 4.8 GB | 4.0 | | | |

**Tokens.** tokenizer.json's byte-level BPE: added tokens (`<|im_start|>`, `<think>`, ...) are matched in the text
first, the stretches between them split by the pre-tokenizer's pattern over the Unicode letter and number classes -
Qwen2's (one digit a chunk), Llama 3's (up to three) or GPT-2's (ByteLevel's own regex: Granite, SmolLM), after
SmolLM's Digits pre-tokenizer where there is one - and each chunk's bytes merged lowest rank first; where the model sets
`ignore_merges` (Llama 3), a chunk the vocabulary holds whole is that token. A byte a vocabulary lacks (SmolLM's and
Granite's lack some control bytes and bytes UTF-8 never has) is dropped, as the library drops it. SentencePiece's BPE
(Gemma's tokenizer.json; a Metaspace pre-tokenizer reads the same way) writes spaces as "▁", does not split the text,
starts from its characters - one the vocabulary lacks as its UTF-8 bytes' `<0xXX>` tokens - and merges them by rank. It encodes as the tokenizers library does
(checked on text in English, German, Chinese, Japanese, Korean, Russian, code, emoji and odd whitespace); the NFC
normalizer is not applied, which matters only for text not already in NFC.

## Checks

`llm/tools/reference.py` builds a small model of random weights for each architecture and option (Llama 3's and YaRN's
rope scaling, Granite's multipliers, SmolLM3's layers without rotation, Mistral's window of 50 over 150 positions, the
mixtures of Qwen3-MoE (a dense layer among them), Granite-MoE and OLMoE) and
writes the logits transformers gives at every position; with `--real DIR ...` it writes a real checkpoint's last-position
logits and ten greedy tokens. `llm/tools/tokref.py DIR ...` writes the tokenizers library's ids of fixed texts (nine
languages and scripts, code, numbers, emoji, whitespace) and the chat template's prompt. The tests read them where
`$OANN_MODELS` has them: every position of every small model within 1e-4 on the exact path (on AMX's, BF16 activations,
where a router may take another expert at a near tie: the mean difference within 1% and the same greedy token at 90% of
positions); SmolLM2-135M, Llama-3.2-1B, Granite-3.3-2B, Granite-3.1-1B-A400M and Gemma-3 270M and 1B (read after `<bos>`) within 1e-3 of the largest logit over
the whole vocabulary and the same ten greedy tokens on the exact path, five on AMX's (Granite-3.3's sixth is a near tie
that transformers in BF16 also takes the other way); the ids of Qwen3, Llama 3.2, SmolLM2, Granite 3.3 and Gemma 3's tokenizers exactly the library's; the chat
prompts exactly the templates'.

A GGUF file's weights load widened to F32 and then rounded to BF16 (or requantized to INT8 with `--int8`): the file's
4 or 8 bits save disk and download, not memory or time. The query and key rows of the Llama family come out of
llama.cpp's interleaved order, and Llama 3's `rope_freqs` divides the frequencies. `llm/tools/ggufref.py FILE.gguf ...`
writes transformers' own reading of each file (dequantized, F32) and gguf-py's dequantization of two rows of a tensor of
each type; the tests hold the rows to 1e-5 and the model, on both paths, to a mean logit difference under 1% of the
largest and the same five greedy tokens (Qwen3-0.6B in Q8_0, Q4_0 with Q4_1 and Q6_K, Q4_K_M, Q5_K_M; Llama-3.2-1B in
Q4_K_M).

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

**Long contexts.** Tokens a second at 2048, 4096 and 8192 positions (BF16 weights; "-" where weights and cache do not
fit in 15 GB):

| model | batch | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1 | 6.7 | 6.9 | 6.1 |
| | 16 | 84 | 74 | 55 |
| | 64 | 185 | 122 (134 with `--kv8`) | - (88 with `--kv8`) |
| Qwen3-4B | 1 | 2.7 | 2.2 | 2.5 |
| | 16 | 28 | - | - |

Past a few thousand positions attention reads more than the products: each position costs 28 KB of cache a step for
the 1.5B model (2 key-value heads of 28 layers), 147 KB for Qwen3-4B (8 of 36), and at batch 16 and more attention
runs at about the machine's bandwidth (25 GB/s). An item of attention (a row and a head group) of more than 16 pages
is split into runs of 16 pages over the tasks and their partial softmaxes joined, so a single long sequence keeps every
core busy (1.5B, batch 1, 8192 positions: attention 24 -> 14 ms a step); where an item splits depends on its own length
only, so a completion stays independent of the batch. Against the exact path over a 1500-token text the split and
unsplit paths agree alike (mean KL 3.6e-5 and 3.3e-5, the same argmax at 99.8% of positions); greedy completions of
1500 tokens were the same as the unsplit build's until the split engaged, and 2 of 4 then took another turn at a near
tie, as two BF16 computations do.

`--kv8` keeps the values in 8 bits with a scale per position and head (the keys stay BF16: a few of their dimensions are
far larger than the rest, and 8 bits per position lost too much - KL 3e-2): 0.76 of the cache, mean KL 5e-5 against the
exact path. It is mostly room - more or longer sequences - and a little speed.

A prompt is read in the same steps, each of its positions a row with attention of its own; that suits prompts of
hundreds of tokens, not of many thousands.

## Models and licences

The weights are fetched from Hugging Face by whoever runs the command; none are in this repository. Models of these
architectures known to run, and their licences (each model's own LICENSE file is fetched beside it):

| model | architecture | licence |
|---|---|---|
| meta-llama/Llama-3.2-1B-Instruct, 3B-Instruct (ungated copies: unsloth/...) | llama | Llama 3.2 Community License (use policy; "Built with Llama") |
| HuggingFaceTB/SmolLM2-135M-Instruct, 360M-Instruct, 1.7B-Instruct | llama | Apache 2.0 |
| HuggingFaceTB/SmolLM3-3B | smollm3 | Apache 2.0 |
| ibm-granite/granite-3.3-2b-instruct, granite-3.3-8b-instruct | granite | Apache 2.0 |
| ibm-granite/granite-3.1-1b-a400m-instruct, granite-3.1-3b-a800m-instruct | granitemoe | Apache 2.0 |
| google/gemma-3-270m-it, gemma-3-1b-it (ungated copies: unsloth/...) | gemma3_text | Gemma Terms of Use (use restrictions; not an open-source licence) |
| allenai/OLMoE-1B-7B-0125-Instruct | olmoe | Apache 2.0 (14 GB in BF16: INT8 on 16 GB) |
| Qwen/Qwen3-30B-A3B | qwen3_moe | Apache 2.0 (61 GB in BF16: does not fit a 16 GB machine yet) |
| mistralai/Mistral-7B-Instruct-v0.3 | mistral | Apache 2.0 (its tokenizer is SentencePiece's: not read yet) |
| Qwen/Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B | qwen3 | Apache 2.0 |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek-R1-Distill-Qwen-7B | qwen2 | MIT (built from Qwen2.5, Apache 2.0) |
| Qwen/Qwen2.5-0.5B-Instruct ... 7B-Instruct | qwen2 | Apache 2.0 |

Check a model's licence before using what it generates.
