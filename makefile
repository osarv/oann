# oann - neural networks in olang. Each .olang file is one module (olang M22); the compiler writes everything it
# builds under build/, and MNIST is cached under data/ (both ignored by git).

# the compiler: olang's build/olang (its makefile's "make"), olang checked out beside oann by default
OLANG ?= $(if $(wildcard ../olang/build/olang),../olang/build/olang,olang)

# every module with test blocks
TESTS = tensor.olang kernels/device.olang kernels/mat.olang kernels/gemm.olang kernels/elem.olang kernels/attention.olang kernels/scan.olang kernels/sparse.olang kernels/conv.olang kernels/fpmode.olang kernels/ternary.olang kernels/nf4.olang kernels/spike.olang \
	ad/graph.olang ad/descent.olang ad/score.olang tools/gradcheck.olang nn/layers.olang nn/moe.olang nn/ngram.olang nn/optim.olang nn/train.olang nn/tune.olang nn/qlora.olang nn/dist.olang nn/generate.olang nn/quant.olang nn/quantgraph.olang nn/message.olang nn/history.olang nn/bestof.olang nn/ternary.olang nn/potts.olang nn/seqsettle.olang nn/diffusion.olang nn/chain.olang nn/lines.olang \
	models/mlp.olang models/cnn.olang models/gpt.olang models/ssm.olang models/qwen.olang tokenizer.olang \
	data/idx.olang data/loader.olang data/mnist.olang data/text.olang data/tokens.olang data/chat.olang io/checkpoint.olang io/onnx.olang io/runlog.olang \
	dyn/neuron.olang dyn/conn.olang dyn/system.olang dyn/energy.olang dyn/learn.olang dyn/holo.olang dyn/store.olang dyn/recur.olang dyn/sleep.olang dyn/halt.olang dyn/bridge.olang models/hybrid.olang board/engine.olang \
	bayes/dense.olang bayes/gauss.olang bayes/quadrature.olang bayes/laplace.olang bayes/state.olang bayes/target.olang bayes/vi.olang \
	bayes/mcmc.olang bayes/empirical.olang bayes/check.olang \
	evo/schema.olang evo/neat.olang evo/pareto.olang evo/search.olang evo/inherit.olang \
	archnet/encode.olang archnet/offline.olang archnet/toy.olang archnet/predict.olang archnet/propose.olang archnet/archnet.olang \
	surgery/model.olang surgery/ops.olang \
	llm/bpe.olang llm/jinja.olang llm/model.olang llm/engine.olang llm/chat.olang llm/gguf.olang llm/vision.olang \
	speech/mel.olang speech/whisper.olang speech/transcribe.olang speech/kws.olang speech/speak.olang speech/vits.olang speech/g2p.olang

# -define Gpu=true wherever the CUDA toolkit's headers are (docs/redesign.md 1.10): every build and test of this makefile
# then has the GPU, and a program uses it where it plans on it (device.SetDefault, or a plan's on). Without the headers
# nothing translates a CUDA header and everything builds for the CPU alone. GPU=0 leaves it out on a machine that has them
GPUDEFINE = $(if $(filter 0,$(GPU)),,$(if $(wildcard /usr/include/cuda_runtime_api.h),-define Gpu=true,))

# the GPU backend's tests: the CPU's tests run green either way (they plan on the CPU); these need the define
GPUTESTS = kernels/gpu/device.olang kernels/gpu/blas.olang kernels/gpu/kern.olang dyn/gpu_test.olang ad/gpu_test.olang

# where WSL keeps its libcuda.so.1, put first on the loader's path for the GPU targets: a native Linux driver package
# installed beside the toolkit (libnvidia-compute) can shadow WSL's in the loader's cache, and CUDA then finds no
# device. Harmless where the system is right or is not WSL (no such directory, nothing set)
GPUENV = $(if $(wildcard /usr/lib/wsl/lib),LD_LIBRARY_PATH=/usr/lib/wsl/lib$(if $(LD_LIBRARY_PATH),:$(LD_LIBRARY_PATH)),)

.DEFAULT_GOAL := help

.PHONY: help diloco oanngen visionbench oannlisten speechbench whisperlinbench vitsbench llmbench grow sparsetable evolm evomlp archsearch potts_latin energybench lmstep lmeq test gputest gpuad gpulm gpumoe bigvocab gpugemm gpurate gpult gpusettle gpulearn mlpad arenaad tinystories data bench epoch mnist cnn charlm lmbench lmref bpe bpelm qwenlora qlora board int8 int8lm int8bench storagebench bf16bench int8cnnbench int8decodebench graphcount historyencode bestof graphsettle samplebench sweepbench flashbench lmlayersbench ternarybench pottsbench focusbench pottsfitbench xor spiking prune hybridbench adjointbench sleepreplay mnistlearn sample clean

# prints this list: every target with the comment above it (its ARGS included) - the default, runs nothing
help:
	@awk '/^#/ { sub(/^# ?/, ""); c = c (c == "" ? "" : " ") $$0; next } \
	  /^[a-z][a-z0-9 ]*:/ { if (c != "") { n = $$0; sub(/:.*/, "", n); printf "%-16s %s\n", n, c } } \
	  { c = "" }' $(MAKEFILE_LIST)

# every module's tests (takes minutes), on the CPU (with the GPU built in where CUDA is, GPUDEFINE) - OLANG=path names
# another compiler
test:
	$(GPUENV) $(OLANG) test $(GPUDEFINE) $(TESTS)

# the GPU backend's tests, CUDA and cuBLAS against the CPU kernels (-define Gpu=true; each test says it is skipped where
# CUDA finds no device) - needs the CUDA toolkit's headers and libraries (cudart, cublas, cublasLt) and an NVIDIA GPU
gputest:
	$(GPUENV) $(OLANG) test -define Gpu=true $(GPUTESTS)

# G-M1: GEMM GFLOPS of kernels/gemm (1 and 4 threads) against cuBLAS on the MLP's and the settle's shapes, F32 and BF16,
# interleaved, medians (bench/gpugemm.olang) - ARGS: rounds, and the CPU's thread counts after it
gpugemm:
	$(OLANG) build -define Gpu=true -output build/bench_gpugemm bench/gpugemm.olang
	$(GPUENV) ./build/bench_gpugemm $(ARGS)

# G-M3: the settle's rate step fused (one launch) against unfused (three), and a sweep of the 784-128-10 settle on the
# GPU - its launches and their share of it, NVRTC's compile and the PTX cache (bench/gpurate.olang) - ARGS: rounds
gpurate:
	$(OLANG) build -define Gpu=true -output build/bench_gpurate bench/gpurate.olang
	$(GPUENV) ./build/bench_gpurate $(ARGS)

# G-M1 on cuBLASLt (GP3): planned products F32, BF16 exact and fast, F32 through BF16 and TF32, BF16 weights under F32,
# against GP1's cuBLAS Gemm and kernels/gemm, resident, with each way's error (bench/gpult.olang) - ARGS: rounds
gpult:
	$(OLANG) build -define Gpu=true -output build/bench_gpult bench/gpult.olang
	$(GPUENV) ./build/bench_gpult $(ARGS)

# GP4's first settle and lesson times of the 784-128-10 system, CPU (1 and 4 threads) against GPU at batch 32, 128 and
# 512, interleaved, medians (bench/gpusettle.olang) - ARGS: rounds, cold settles and lessons a round, Check
gpusettle:
	$(OLANG) build -define Gpu=true -output build/bench_gpusettle bench/gpusettle.olang
	$(GPUENV) ./build/bench_gpusettle $(ARGS)

# G-M2: examples/mnist_learn on MNIST, the settle on the CPU (1 and 4 threads) against the GPU (Check 1, 4 and 16) at
# batch 32, 128, 512 and hidden 128, 1024 - epoch time, sweeps a lesson, test accuracy; interleaved, medians
# (bench/gpulearn.olang; MNIST from make data) - ARGS: rounds, epochs, then hidden sizes, batches and backends as lists
gpulearn:
	$(OLANG) build -define Gpu=true -output build/examples_mnist_learn examples/mnist_learn.olang
	$(OLANG) build -define Gpu=true -output build/bench_gpulearn bench/gpulearn.olang
	$(GPUENV) ./build/bench_gpulearn $(ARGS)

# G-M4: ad/graph on the GPU - M1's perceptron epoch (MNIST, make data) and M2's character-transformer step on the CPU (1 and
# 4 threads) and the GPU (AdamW and the clip on the device), interleaved, medians (bench/gpuad.olang) - ARGS: rounds,
# transformer steps a round, and mlp or lm to run one only
gpuad:
	$(OLANG) build -define Gpu=true -output build/bench_gpuad bench/gpuad.olang
	$(GPUENV) ./build/bench_gpuad $(ARGS)

# M1 on ad/graph: the 784-128-10 perceptron on MNIST, each epoch timed - ARGS: epochs, threads, seed, and bf16
mlpad:
	$(OLANG) build $(GPUDEFINE) -output build/bench_mlp_ad bench/mlp_ad.olang
	$(GPUENV) ./build/bench_mlp_ad $(ARGS)

# M9 on ad/graph: round 2's transformer's arena with activations reused and apart, and the resident memory per step -
# ARGS: steps, threads, and bf16
arenaad:
	$(OLANG) build $(GPUDEFINE) -output build/bench_arena_ad bench/arena_ad.olang
	$(GPUENV) ./build/bench_arena_ad $(ARGS)

# G-M6: what language model fits on the GPU and how fast it trains - ARGS: dim layers heads ctx batch steps f32|bf16 [vocab]
gpulm:
	$(OLANG) build $(GPUDEFINE) -output build/bench_gpulm bench/gpulm.olang
	$(GPUENV) ./build/bench_gpulm $(ARGS)

# a mixture-of-experts layer (nn/moe on the device) against the dense gated perceptron of its active size: median steps
# and the mixture's cost over the dense layer's (bench/gpumoe.olang) - ARGS: tokens steps f32|bf16 experts k D...
gpumoe:
	$(OLANG) build $(GPUDEFINE) -output build/bench_gpumoe bench/gpumoe.olang
	$(GPUENV) ./build/bench_gpumoe $(ARGS)

# the activation arena of a BF16 model at Qwen3's sizes: full cross entropy, ChunkedCE, ChunkedCE with Recompute
bigvocab:
	$(OLANG) build -output build/bench_bigvocab bench/bigvocab.olang
	./build/bench_bigvocab $(ARGS)

# fetches MNIST into data/mnist (once), checks the pipeline against what is known about the dataset, and times it
data bench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_data bench/data.olang
	$(GPUENV) ./build/bench_data

# trains the 784-128-10 perceptron on MNIST and measures it on the test set after every epoch - extra arguments go
# through ARGS: epochs, optimizer (adamw, projected, sgd), seed, threads
mnist:
	$(OLANG) build $(GPUDEFINE) -output build/examples_mnist_mlp examples/mnist_mlp.olang
	$(GPUENV) ./build/examples_mnist_mlp $(ARGS)

# trains a small convolutional network on MNIST (two 3x3 convolutions with ReLU and 2x2 max pooling, then a dense layer)
# - ARGS: epochs, seed, threads, the convolutions' channels C1 and C2, and "profile" for where the first epoch goes
cnn:
	$(OLANG) build $(GPUDEFINE) -output build/examples_mnist_cnn examples/mnist_cnn.olang
	$(GPUENV) ./build/examples_mnist_cnn $(ARGS)

# DiLoCo (nn/dist): the character transformer trained by one worker, by data parallel or by N DiLoCo worker processes
# through a directory store, the validation loss against tokens and seconds (examples/diloco_lm.olang) - ARGS: "single
# STEPS", "dp R STEPS" or "diloco N STEPS H [CODEC] [FRAGMENTS] [DIR]"
diloco:
	$(OLANG) build -output build/examples_diloco examples/diloco_lm.olang
	./build/examples_diloco $(ARGS)

# M2 and M3: the character-level transformer (models/gpt) on tiny Shakespeare (fetched into data/shakespeare once)
# through nn/train's Fit, its checkpoint in data/shakespeare, and a sample through the key-value cache - ARGS: steps,
# seed, threads, dropout, characters to generate, and bf16 last for mixed precision
charlm:
	$(OLANG) build $(GPUDEFINE) -output build/examples_charlm examples/charlm.olang
	$(GPUENV) ./build/examples_charlm $(ARGS)

# M13: a GPT and an SSM of 10 million parameters on TinyStories (examples/tinystories.olang) - first ARGS="prep", then
# ARGS="gpt TOKENS" or "mamba TOKENS" (context, sequences, seed, threads, bf16 after)
tinystories:
	$(OLANG) build $(GPUDEFINE) -output build/examples_tinystories examples/tinystories.olang
	$(GPUENV) ./build/examples_tinystories $(ARGS)

# the same transformer on byte-level BPE tokens (512, learned from the training part), its loss per character against
# the character model's checkpoint - ARGS: steps, seed, threads, vocabulary, tokens to generate
bpelm:
	$(OLANG) build $(GPUDEFINE) -output build/examples_bpelm examples/bpelm.olang
	$(GPUENV) ./build/examples_bpelm $(ARGS)

# LoRA on Qwen3 on the CPU (examples/qwen_lora.olang) - ARGS: the model's directory, steps, adapter and merged directories
qwenlora:
	$(OLANG) build $(GPUDEFINE) -output build/examples_qwen_lora examples/qwen_lora.olang
	$(GPUENV) ./build/examples_qwen_lora $(ARGS)

# QLoRA fine-tuning of Qwen3 on chat conversations (examples/qlora.olang, nn/qlora; GPU where there is one) - ARGS: the
# model's directory, the training JSONL, the output directory, then "-name value" options (-steps, -ctx, -rank, ...), or
# -plan for what a run would take
qlora:
	$(OLANG) build $(GPUDEFINE) -output build/examples_qlora examples/qlora.olang
	$(GPUENV) ./build/examples_qlora $(ARGS)

# oann-listen: speech to text by a Whisper model (speech/listen.olang, speech/README.md), built to build/oann-listen -
# ARGS: MODEL CLIP.wav and its flags (none: only builds it)
oannlisten:
	$(OLANG) build -output build/oann-listen speech/listen.olang
	$(if $(ARGS),./build/oann-listen $(ARGS),)

# oann-gen: completions of a JSON-lines file of prompts by an open-weight model (llm/gen.olang, llm/README.md), built
# to build/oann-gen - ARGS: MODEL prompts.jsonl out.jsonl and its flags (none: only builds it)
oanngen:
	$(OLANG) build -output build/oann-gen llm/gen.olang
	$(if $(ARGS),./build/oann-gen $(ARGS),)

# Whisper's real-time factor (bench/speech.olang) - ARGS: MODEL_DIR CLIP.wav [rounds] [threads ...]
speechbench:
	$(OLANG) build -output build/bench_speech bench/speech.olang
	./build/bench_speech $(ARGS)

# speech/whisper's few-row products (Arith.Lin) at Whisper base's decoder shapes, F32 and INT8, on each thread count
# (bench/whisperlin.olang) - ARGS: [rounds] [rows] [threads ...]
whisperlinbench:
	$(OLANG) build -output build/bench_whisperlin bench/whisperlin.olang
	./build/bench_whisperlin $(ARGS)

# VITS text to speech (bench/vits.olang): seconds of compute a second of audio - ARGS: a VitsModel directory or a
# Piper voice's .onnx, [rounds] [threads ...]
vitsbench:
	$(OLANG) build -output build/bench_vits bench/vits.olang
	./build/bench_vits $(ARGS)

# decoding speed of an open-weight model, tokens a second at batch sizes, and peak memory (bench/llm.olang) - ARGS: the
# model's directory, context, steps, int8 or bf16, batch sizes
visionbench:
	$(OLANG) build -output build/bench_vision bench/vision.olang
	./build/bench_vision $(ARGS)

llmbench:
	$(OLANG) build -output build/bench_llm bench/llm.olang
	./build/bench_llm $(ARGS)

# where a transformer's training step goes, operation by operation - ARGS: steps, threads, layers, width, heads,
# context, sequences
lmbench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_lm bench/lm.olang
	$(GPUENV) ./build/bench_lm $(ARGS)

# the first steps' losses against the same model in numpy (bench/ref/charlm.py) - ARGS: steps
lmref:
	$(OLANG) build $(GPUDEFINE) -output build/bench_lmref bench/lmref.olang
	$(GPUENV) ./build/bench_lmref $(ARGS) > build/lmref_olang.txt
	python3 -I bench/ref/charlm.py $(ARGS) > build/lmref_numpy.txt
	paste build/lmref_olang.txt build/lmref_numpy.txt

# byte-level BPE on tiny Shakespeare: training and encoding timed, and the merges and tokens checked against an
# independent Python implementation (bench/ref/bpe.py) - ARGS: the vocabulary
bpe:
	$(OLANG) build $(GPUDEFINE) -output build/bench_bpe bench/bpe.olang
	$(GPUENV) ./build/bench_bpe $(or $(ARGS),512) build/bpe_olang.txt
	python3 -I bench/ref/bpe.py data/shakespeare/input.txt $(or $(ARGS),512) build/bpe_olang.txt

# MNIST by a settling system the board can run, its lessons settled on the simulated board engine (board/engine) or in
# F32, the test set answered both ways after every epoch - ARGS: epochs, board or float, samples, hidden, batch, seed,
# rate, beta, temperature
board:
	$(OLANG) build $(GPUDEFINE) -output build/examples_mnist_board examples/mnist_board.olang
	$(GPUENV) ./build/examples_mnist_board $(ARGS)

# the exact implicit gradient's adjoint, plain sweeps against Anderson's at depth 3 and 5, on 2- and 3-layer rate
# systems: sweeps and time (bench/adjoint.olang) - ARGS: rounds, batch, tolerance, seed
adjointbench:
	$(OLANG) build -output build/bench_adjoint bench/adjoint.olang
	./build/bench_adjoint $(ARGS)

# M14 (docs/redesign.md 4.3): models/hybrid's transformer with a settle block on tiny Shakespeare, trained twice - every
# settle from rest, then from the amortizer's guess - and the sweeps a settle takes in each (bench/hybrid.olang) - ARGS:
# steps, seed, units, gradient (exact, jfb, ep), tolerance, kappa, report interval
hybridbench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_hybrid bench/hybrid.olang
	$(GPUENV) ./build/bench_hybrid $(ARGS)

# message passing over random graphs, sum against mean aggregation: a counting and a majority node task, tested on
# small and on larger graphs (examples/graph_count.olang) - ARGS: steps, seed
graphcount:
	$(OLANG) build $(GPUDEFINE) -output build/examples_graph_count examples/graph_count.olang
	./build/examples_graph_count $(ARGS)

# a one-block self-attention encoder over variable-length histories against a larger flat perceptron
# (examples/history_encode.olang) - ARGS: steps, seed
historyencode:
	$(OLANG) build $(GPUDEFINE) -output build/examples_history_encode examples/history_encode.olang
	./build/examples_history_encode $(ARGS)

# best-of-N by energy on a continuous Hopfield memory: one settle against the lowest energy of N against any of N
# (examples/best_of.olang) - ARGS: patterns, flipped fraction, drive, gain, batches, seed
# nn/potts end to end on Latin squares: learnt, then solved by Anneal and Focus (examples/potts_latin.olang) - ARGS: n,
# examples, puzzles, clue share, seed
# the energy toolkit on generic puzzles, one thread: examples/potts_latin and examples/energy_suite (Focus on 3-SAT and
# colouring, CutsetMove on grid distances, BySum cage sums, nn/lines run counts) - accuracy, time, moves
# a training step of a small BPE language model on the CPU: per-operation times, tokens a second, threads inside the
# operations or the batch split over replicas (bench/lmstep.olang) - ARGS: steps threads blocks width heads window batch vocab [dp]
# F32 against BF16 training of bench/lmstep's model on TinyStories, SECONDS each (bench/lmeq.olang; after
# examples/tinystories prep) - ARGS: f32|bf16 seconds seed
lmeq:
	$(OLANG) build -output build/bench_lmeq bench/lmeq.olang
	./build/bench_lmeq $(ARGS)

lmstep:
	$(OLANG) build -output build/bench_lmstep bench/lmstep.olang
	./build/bench_lmstep $(ARGS)

energybench:
	$(OLANG) build -output build/examples_potts_latin examples/potts_latin.olang
	./build/examples_potts_latin
	$(OLANG) build -output build/examples_energy_suite examples/energy_suite.olang
	./build/examples_energy_suite

potts_latin:
	$(OLANG) build -output build/examples_potts_latin examples/potts_latin.olang
	./build/examples_potts_latin $(ARGS)

bestof:
	$(OLANG) build $(GPUDEFINE) -output build/examples_best_of examples/best_of.olang
	./build/examples_best_of $(ARGS)

# a settle step on graph-structured states: the dense adjacency product against message passing over its edges, and the
# state's gradient alone (bench/graphsettle.olang) - ARGS: steps, batch, D, hidden, node counts
# the settle's sweep cost: a self-coupled hidden population of H units at batch B, its weights packed in every sweep
# against packed once a settle (the reciprocal self-connection then one product of W^T + W) (bench/sweep.olang) - ARGS:
# sweeps, threads, hidden sizes
sweepbench:
	$(OLANG) build -output build/bench_sweep bench/sweep.olang
	./build/bench_sweep $(ARGS)

# attention keeping its weights against the flash form keeping none: forward and backward ms, causal, F32, contexts 256
# to 4096 (bench/flash.olang) - ARGS: threads, rounds
flashbench:
	$(OLANG) build -output build/bench_flash bench/flash.olang
	./build/bench_flash $(ARGS)

# the language-model layers on one thread, tokens/s forward and backward at width 160 and contexts of 2048: sliding-window
# attention (a scalar loop, the flash form F32 and BF16), the delta rule, n-gram tables (bench/lmlayers.olang) - ARGS: rounds
lmlayersbench:
	$(OLANG) build -output build/bench_lmlayers bench/lmlayers.olang
	./build/bench_lmlayers $(ARGS)

# kernels/ternary's matrix-vector product (2-bit weights) against gemm's F32 Gemv on the same weights, one thread
# (bench/ternary.olang) - ARGS: rounds
ternarybench:
	$(OLANG) build -output build/bench_ternary bench/ternary.olang
	./build/bench_ternary $(ARGS)

# nn/potts: annealed chromatic Gibbs over many chains on planted 3-colourable graphs of 256 to 2048 cells, 1 and 4
# threads (bench/potts.olang) - ARGS: chains, sweeps
pottsbench:
	$(OLANG) build -output build/bench_potts bench/potts.olang
	./build/bench_potts $(ARGS)

# nn/potts' focused heat bath (Sampler.Focus) on planted 3-SAT and 3-colouring, plain and with breakout weights: ns a
# move, chains solved, a checksum of the answers (bench/focus.olang) - ARGS: chains, moves, threads
focusbench:
	$(OLANG) build -output build/bench_focus bench/focus.olang
	./build/bench_focus $(ARGS)

# nn/potts' fitters - pseudolikelihood, CD-1, persistent CD-1, EP - on planted colourings with clues, then annealed
# search on new graphs of 128 and 256 cells (bench/pottsfit.olang) - ARGS: updates, clues' share, test graphs, seed
pottsfitbench:
	$(OLANG) build -output build/bench_pottsfit bench/pottsfit.olang
	./build/bench_pottsfit $(ARGS)

graphsettle:
	$(OLANG) build $(GPUDEFINE) -output build/bench_graphsettle bench/graphsettle.olang
	./build/bench_graphsettle $(ARGS)

# sampling a batch of continuations: rerunning the context for every token against batched key-value-cached decoding
# (bench/sample.olang) - ARGS: layers, width, batch, prompt, new tokens, vocabulary
samplebench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_sample bench/sample.olang
	./build/bench_sample $(ARGS)

# growing a trained small LM into a larger one against training the larger from scratch, at equal compute, on tiny
# Shakespeare (bench/grow.olang) - ARGS: budget in large-model steps, the small model's share, seed, threads
grow:
	$(OLANG) build $(GPUDEFINE) -output build/bench_grow bench/grow.olang
	./build/bench_grow $(ARGS)

# a 262144 x 160 lookup table trained densely and sparsely (Graph.Sparse, AdamW.Sparse) at batch 32 x 128
# (bench/sparsetable.olang) - ARGS: steps, rows, width, batch, context
sparsetable:
	$(OLANG) build $(GPUDEFINE) -output build/bench_sparsetable bench/sparsetable.olang
	./build/bench_sparsetable $(ARGS)

# archnet on its toy benchmark: the predictor's held-out rank correlation, and evolution alone against screening,
# proposals and forecast halving (bench/archsearch.olang) - ARGS: rounds, seeds, gap
archsearch:
	$(OLANG) build $(GPUDEFINE) -output build/bench_archsearch bench/archsearch.olang
	./build/bench_archsearch $(ARGS)

# archnet's predictor measured offline on a log of finished runs (loss, curve, features; archnet/offline.olang): Kendall
# tau from the first m curve points, cross-validated (bench/archeval.olang) - ARGS: log, m list (0,1,2,5,10), folds, seed
archeval:
	$(OLANG) build $(GPUDEFINE) -output build/bench_archeval bench/archeval.olang
	./build/bench_archeval $(ARGS)

# architecture search over tiny character LMs (depth, width, heads, window, ffn, norm, rate) on tiny Shakespeare with
# evo, successive halving over steps (examples/evo_lm.olang) - ARGS: rounds, workers, seed, directory
evolm:
	$(OLANG) build $(GPUDEFINE) -output build/examples_evo_lm examples/evo_lm.olang
	./build/examples_evo_lm $(ARGS)

# architecture search with evo: regularized evolution over MLPs for a synthetic regression, successive halving, weight
# inheritance, checkpoints, worker processes (examples/evo_mlp.olang) - ARGS: rounds, workers, seed, directory
evomlp:
	$(OLANG) build $(GPUDEFINE) -output build/examples_evo_mlp examples/evo_mlp.olang
	./build/examples_evo_mlp $(ARGS)

# XOR by rate neurons on dyn, taught by equilibrium propagation (examples/xor_settle.olang)
xor:
	$(OLANG) build $(GPUDEFINE) -output build/examples_xor_settle examples/xor_settle.olang
	$(GPUENV) ./build/examples_xor_settle $(ARGS)

# a settling system pruned while it runs, then stored sparse (examples/prune_settle.olang) - ARGS: score, fraction kept,
# pace, regrowth, hidden, noise inputs, lessons, lessons between ticks, seed, rate, schedule (linear, cubic), skip (none,
# drawn, zero)
prune:
	$(OLANG) build $(GPUDEFINE) -output build/examples_prune_settle examples/prune_settle.olang
	$(GPUENV) ./build/examples_prune_settle $(ARGS)

# XOR by spiking neurons on dyn (examples/xor_spiking.olang)
spiking:
	$(OLANG) build $(GPUDEFINE) -output build/examples_xor_spiking examples/xor_spiking.olang
	$(GPUENV) ./build/examples_xor_spiking $(ARGS)

# sleep replay on dyn (examples/sleep_replay.olang) - ARGS: seeds, days, patterns a day, hidden, passes, rate, context,
# store cells, active cells
sleepreplay:
	$(OLANG) build $(GPUDEFINE) -output build/examples_sleep_replay examples/sleep_replay.olang
	$(GPUENV) ./build/examples_sleep_replay $(ARGS)

# the same network on dyn (dyn/learn's EP lessons; docs/redesign.md 4.3, M8) - ARGS: epochs, samples, hidden, batch,
# seed, Adam's rate, beta, threads
mnistlearn:
	$(OLANG) build $(GPUDEFINE) -output build/examples_mnist_learn examples/mnist_learn.olang
	$(GPUENV) ./build/examples_mnist_learn $(ARGS)

# text from the checkpoint make charlm wrote (examples/charlm_sample.olang)
sample:
	$(OLANG) build $(GPUDEFINE) -output build/examples_charlm_sample examples/charlm_sample.olang
	$(GPUENV) ./build/examples_charlm_sample $(ARGS)

# the perceptron (or the CNN) trained in F32, quantized to INT8 (nn/quant.olang, M12) and measured both ways: test accuracy,
# the parameters' size, throughput at batch 1, 64 and 128 - ARGS: mlp or cnn, epochs,
# seed, threads, timing rounds
int8:
	$(OLANG) build $(GPUDEFINE) -output build/examples_mnist_int8 examples/mnist_int8.olang
	$(GPUENV) ./build/examples_mnist_int8 $(ARGS)

# the character transformer (data/shakespeare/charlm.ckpt, from make charlm) with its linear layers in INT8
# (nn/quantgraph's Quantize): validation loss, a forward's time, greedy text and cached decoding against F32 - ARGS:
# characters, timing rounds, threads
int8lm:
	$(OLANG) build $(GPUDEFINE) -output build/examples_charlm_int8 examples/charlm_int8.olang
	$(GPUENV) ./build/examples_charlm_int8 $(ARGS)

# the INT8 product against kernels/gemm's F32 product with its bias and ReLU, on oann's networks' shapes (bench/int8.olang)
# - ARGS: rounds, threads
int8bench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_int8 bench/int8.olang
	$(GPUENV) ./build/bench_int8 $(ARGS)

# BF16 x BF16 products into F32 widened, by the pair dot and by AMX tiles, one thread (bench/bf16gemm.olang) - ARGS: rounds
bf16bench:
	$(OLANG) build -output build/bench_bf16gemm bench/bf16gemm.olang
	./build/bench_bf16gemm $(ARGS)

# F32 activations times weights stored in F32, BF16 or INT8 through one Gemm (bench/storage.olang) - ARGS: rounds, threads
storagebench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_storage bench/storage.olang
	$(GPUENV) ./build/bench_storage $(ARGS)

# the INT8 CNN of nn/quant against the graph's F32 forward, as drawn, at batch 1, 64, 128 (bench/int8cnn.olang) - ARGS:
# rounds, threads
int8cnnbench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_int8cnn bench/int8cnn.olang
	$(GPUENV) ./build/bench_int8cnn $(ARGS)

# the character GPT decoding through the key-value cache, F32 against INT8 (bench/int8decode.olang) - ARGS: characters,
# rounds, threads, and int8 to time INT8 alone (for olang profile)
int8decodebench:
	$(OLANG) build $(GPUDEFINE) -output build/bench_int8decode bench/int8decode.olang
	$(GPUENV) ./build/bench_int8decode $(ARGS)

# an epoch's time against C over OpenBLAS (bench/epoch.olang)
epoch:
	$(OLANG) build $(GPUDEFINE) -output build/bench_epoch bench/epoch.olang
	$(GPUENV) OLANG=$(OLANG) ./build/bench_epoch

# removes build/
clean:
	rm -rf build
