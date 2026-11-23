# speech

Speech on the CPU, for a voice assistant on a Raspberry Pi and dictation on a phone: speech to text by OpenAI's Whisper
(tiny, base, small; the larger ones load alike), a wake word by a small network one trains oneself, and a survey of
text-to-speech models that would fit a Pi. Nothing here trains Whisper; weights are fetched at run time and never
committed, as in `llm/`.

| file | what |
|---|---|
| `mel.olang` | `Wav` (16-bit PCM WAVE, any rate, resampled to 16 kHz), `Mel` (log-mel: Hann window 400, hop 160, Slaney's filters, the DFT as a matrix product), `Mel.Clip` (Whisper's input, as Hugging Face's feature extractor makes it) |
| `whisper.olang` | `Model.Load` (config.json, model.safetensors), `Encode` (the convolutions and blocks over 1500 positions, or fewer), `Decoder` (cross-attention keys and values, a cache of the decoder's own, beams in slots), `Step` |
| `transcribe.olang` | `Whisper`: the prompt and language detection, the logit rules (suppressed tokens, timestamp rules), greedy and beam decoding, `Cut` (segments), `Transcribe` (clips of any length), `Stream` (sound as it comes) |
| `kws.olang` | the wake word: features, the network (`Spotter`), `Train`, `Detector` (a threshold over smoothed scores, a refractory second) |
| `listen.olang` | `oann-listen`, the command |
| `tools/reference.py` | the reference the tests are held to: transformers, F32, on CPU |

## Using it

```
make oannlisten                                                  # builds build/oann-listen
build/oann-listen openai/whisper-base clip.wav                    # the text
build/oann-listen openai/whisper-base talk.wav --timestamps       # a line a segment: [12.40 - 15.80] text
build/oann-listen openai/whisper-small clip.wav --beams 5 --language sv
arecord -q -f S16_LE -r 16000 -c 1 | build/oann-listen openai/whisper-base -            # dictation
arecord -q -f S16_LE -r 16000 -c 1 | build/oann-listen openai/whisper-base - --wake wake.safetensors   # assistant
```

MODEL is a directory or a Hugging Face repository, fetched once into `$OANN_MODELS` (`~/.cache/oann/models`). Flags:
`--timestamps`, `--beams N`, `--language L` (detected otherwise), `--translate` (into English), `--bf16` (BF16 weights:
half the memory, AMX products where the CPU has them), `--threads T`, `--positions N` (a shorter encoder window: faster
on short commands, not the reference), `--max-tokens`; with `--wake`: `--threshold`, `--pause`, `--longest`, `--quiet`.

For the Pi: `olang build -target aarch64-linux-gnu -output build/oann-listen speech/listen.olang` (clang 19 or newer and
the arm64 C library, olang's docs/install.md), then copy the binary over.

## How it works

**Front end.** The clip is padded with zeros to 30 s (a longer one is taken whole), each 400-sample frame windowed and
transformed by one matrix product against the 402 rows of the DFT's cosines and sines, its power put through 80 mel
filters, log10, clamped to the largest less 8 and scaled, as Hugging Face's feature extractor does (`Mel.Clip`).

**Model.** Two convolutions of width 3 (stride 1, then 2) with GELU, sinusoidal positions, pre-norm blocks of
self-attention and a GELU perceptron; the decoder adds cross-attention to the 1500 encoded positions and its head is the
token embedding. Every linear map and the encoder's attention are kernels/gemm products, split over the threads by
outputs where gemm would not split them (the decoder's few rows; AMX tiles, whose path in gemm is one task). By default
everything is F32 - the reference's arithmetic; `--bf16` keeps the weights in BF16, multiplies the encoder's products on
AMX tiles and the decoder's few rows as F32 activations times the widened weights. GELU and the softmax use
kernels/elem's vectorised exp and erfc (3e-7 relative), their sums in F64 lanes.

**Decoding** is transformers' generate for Whisper, step for step: the prompt `<|startoftranscript|><|lang|>
<|transcribe|>` (and `<|notimestamps|>` without timestamps), the language the likeliest language token after the start
when not given; each step suppresses generation_config's tokens, the first step also a blank and the end; with
timestamps, the timestamp rules (pairs, never backwards, the first within 1 s, a timestamp when the timestamps together
are likelier than any text token). Greedy takes the likeliest token. Beam search is transformers' (length penalty 1,
early stopping off): 2B candidates from all beams' log-probabilities plus scores, the best B that do not end run on
(their caches copied into the other half of the slots), those of the first B that end offered to the finished set by
score over length, until no running beam can beat the worst finished one.

**Long clips and streaming.** `Transcribe` reads window after window from one log-mel of the whole clip: a window's
segments are the text between pairs of timestamps; the next window starts at the last pair, or after the whole window
where it ends in a lone timestamp or has no pairs (OpenAI's and transformers' sequential reading; a window past the
clip's end is zeros in the log-mel). `Stream` takes sound in chunks and settles a window as soon as 30 s of it are
there (each window normalised on its own), `Partial` decodes what is not settled yet; `oann-listen -` prints settled
segments. For commands, the assistant loop takes what follows the wake word until a pause and decodes it as one clip.

## Checks

`make test` runs the module tests; where `$OANN_SPEECH` (default `~/.cache/oann`) has `models/openai--whisper-*`,
`audio/*.wav` and `ref/MODEL/*` (written by `python3 -I speech/tools/reference.py MODEL_DIR CLIP.wav OUT`), they hold
the implementation to transformers 5.24 (F32, CPU):

- the log-mel within 1e-4 of the feature extractor's (its own torch and numpy paths differ by 4.4e-5);
- the decoder's logits over the whole greedy sequence (vocabulary 51865 at every position) within 1e-4: largest gaps
  7.6e-5 (tiny), 5.7e-5 (base), 5.7e-5 (small); against transformers run in F64, 3.4e-5 for tiny (transformers' own F32
  is 7.7e-5 from its F64);
- greedy and 5-beam transcripts token for token on three clips, and the timestamped segments (text and times) of four,
  one of 67 s read in three windows, for all three models;
- BF16 products: logits within 0.31 (tiny), 0.19 (base), 0.14 (small) of F32, the same greedy transcripts.

`OANN_SPEECH_QUICK=1` checks tiny alone. The aarch64 build passes the same tiny checks under qemu-user (below).

The wake-word test trains a network on a synthetic word (a rising chirp and a tone) against noise and decoys (a
falling chirp and another tone), saves and loads it, and finds the three words in 9 s of a stream and not the decoy.

## Speed

On a 4-core cloud VM (Xeon, Sapphire Rapids, 2.1 GHz, AVX-512 and AMX), a 24 s clip (`make speechbench ARGS="DIR
CLIP.wav 3 1 4"`, bench/speech.olang): the encoder's 30 s window, the decoder per token, and the real-time factor of the
clip (log-mel + encoder + greedy decoding of its 75-78 tokens, over 24 s):

| model | params | products | threads | encoder | decoder / token | RTF |
|---|---|---|---|---|---|---|
| tiny | 39 M | F32 | 1 | 0.59 s | 13.9 ms | 0.069 |
| | | F32 | 4 | 0.24 s | 8.4 ms | 0.037 |
| | | BF16 | 1 | 0.38 s | 12.4 ms | 0.055 |
| | | BF16 | 4 | 0.18 s | 6.1 ms | 0.027 |
| base | 74 M | F32 | 1 | 1.31 s | 27 ms | 0.14 |
| | | F32 | 4 | 0.49 s | 18 ms | 0.080 |
| | | BF16 | 4 | 0.62 s | 17 ms | 0.083 |
| small | 244 M | F32 | 1 | 4.58 s | 75 ms | 0.28 * |
| | | F32 | 4 | 1.64 s | 40 ms | 0.12 * |
| | | BF16 | 1 | 2.58 s | 66 ms | 0.19 * |
| | | BF16 | 4 | 0.93 s | 37 ms | 0.083 * |

\* small wrote 28 tokens of this clip (it skips the LibriVox preamble); at 75 tokens its RTF would be about 0.26 (F32,
4 threads).

A window costs one encoder pass whatever the clip's length, so a 3 s command costs about what a 30 s clip does;
`--positions 300` (a 6 s window) cuts the encoder by about five (an approximation: the model saw 30 s windows).

**Raspberry Pi 5, estimated** (not measured): Cortex-A76 at 2.4 GHz, two 128-bit NEON FMA pipes - 38 GFLOPS a core at
peak against about 190 here (two 512-bit FMA units at up to 3 GHz) - and LPDDR4X at about 10 GB/s in practice against
about 25 here. The encoder is bound by arithmetic (about 4x slower), the decoder by reading its weights (about 2.5x at 4
threads, 3x at 1), and there is no AMX, so F32 (or `--bf16` for the memory, widened in software):

| model | threads | encoder (30 s window) | decoder / token | 24 s clip, ~75 tokens: RTF |
|---|---|---|---|---|
| tiny | 4 | ~1.0 s | ~21 ms | ~0.11 |
| tiny | 1 | ~2.4 s | ~42 ms | ~0.23 |
| base | 4 | ~2.0 s | ~45 ms | ~0.23 |
| base | 1 | ~5.3 s | ~81 ms | ~0.48 |
| small | 4 | ~6.5 s | ~100 ms | ~0.6 |
| small | 1 | ~18 s | ~225 ms | ~1.5 |

So tiny and base are faster than real time on a Pi 5 with room to spare; a spoken command of 3 s through base costs
about 2.5 s (encoder 2 s, ten tokens), or under a second with `--positions 300`. small is usable for dictation on 4
threads, not as an instant assistant.

## Text to speech: the survey

Models with open code and permissively licensed weights, for a Pi (sizes are the published files):

| model | architecture | params / size | code | weights | input | fits a Pi 5 |
|---|---|---|---|---|---|---|
| kakao-enterprise/vits-ljs | VITS (end to end, HiFi-GAN decoder) | 36 M, 145 MB F32 (safetensors, transformers' `VitsModel`) | MIT | MIT; LJ Speech, public domain | IPA phonemes (espeak-ng's) | yes: VITS runs in real time on a Pi 4 (Piper) |
| rhasspy/piper-voices (e.g. en_US-ljspeech-medium) | VITS, Piper's training | 15-30 M, 60-75 MB a voice (ONNX) | MIT (rhasspy/piper; the successor piper1-gpl is GPL-3) | MIT repository; each voice's dataset licence in its MODEL_CARD (LJ Speech: public domain; some voices non-commercial) | espeak-ng phonemes | yes, the reference for Pi TTS |
| KittenML/kitten-tts-nano-0.1 | StyleTTS-like, ONNX | 15 M, 25 MB | Apache 2.0 | Apache 2.0 | espeak-ng phonemes | yes |
| hexgrad/Kokoro-82M | StyleTTS 2 + iSTFTNet | 82 M, 330 MB | Apache 2.0 | Apache 2.0 | misaki G2P (Apache) or espeak-ng | about real time on 4 cores; best quality here |
| myshell-ai/MeloTTS-English | VITS 2-like + BERT features | ~50 M + BERT | MIT | MIT | g2p_en (CMUdict) + BERT | heavier (the BERT) |
| facebook/mms-tts-eng | VITS | 36 M | MIT (transformers) | **CC BY-NC 4.0**: not permissive | characters | excluded |
| Coqui XTTS-v2 | GPT + HiFi-GAN | 470 M | MPL | **CPML, non-commercial** | text | excluded |

The best fit is VITS on LJ Speech: kakao-enterprise/vits-ljs is MIT end to end on public-domain data, small (36 M), in
safetensors and in transformers (so it can be held to a reference as Whisper is), and its architecture is Piper's, so
Piper's MIT voices can follow. What every candidate but MMS needs is a grapheme-to-phoneme step - espeak-ng is GPL-3, so
it would be our own: CMUdict (BSD-2) for known words and letter-to-sound rules for the rest, mapped to the model's IPA
symbols. Not built yet: the network is clearly small enough (about 1 GFLOP a second of speech), the G2P is the work.

## Models, data and licences

Weights and clips are fetched by whoever runs the code; none are in this repository.

| what | used for | licence |
|---|---|---|
| openai/whisper-tiny, -base, -small (Hugging Face) | speech to text | Apache 2.0 on the Hub; OpenAI's release (github.com/openai/whisper, code and weights) MIT |
| jfk.wav (JFK's inaugural address, 1961, as whisper.cpp's sample) | test clip | public domain (a work of the US government) |
| LibriVox, Aesop's Fables vol. 1, sections 1-2 (archive.org `aesop_fables_volume_one_librivox`), cut to 24, 20 and 67 s | test clips | public domain (LibriVox) |
| the synthetic word of the kws test | wake-word test | made by the test |
| transformers, torch (reference.py only) | the reference | Apache 2.0, BSD-3; not vendored, not needed to run |

The clips are made by `ffmpeg -i IN.mp3 -ss S -t T -ar 16000 -ac 1 -c:a pcm_s16le OUT.wav` from
`https://archive.org/download/aesop_fables_volume_one_librivox/fables_01_0{1,2}_aesop_64kb.mp3` (jfk.wav from
`https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav`).

## The wake word: data

The network learns from whatever clips it is given (`kws.Train(config, positives, negatives, options)`): a few hundred
recordings of the word by the people who will say it (a second or two each, any room, any microphone), and negatives -
other speech, the room's noise, music, silence - as long as there is. Synthesised positives (a TTS voice saying the
word, at varied speeds and pitches) fill out a small set. Augmentation is built in: each positive is shifted up to 0.2
s and mixed with a negative at a random level, and half the negatives are centred on their loudest stretch so that
other words are seen where the word would be. A model is about 10 000 parameters (40 KB); scoring a window costs well
under a millisecond, so on a Pi the detector runs every 100 ms on one core with room to spare.
