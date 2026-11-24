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
| `speak.olang` | `Voice.Load("en" or "sv")`, `Speak(text)`: our G2P (`g2p.olang`), then a Piper voice (`vits.olang`), samples at `Rate` (22050); `To16k` |
| `listen.olang` | `oann-listen`, the command |
| `g2p.olang` | `G2p.Load`, `Phonemes(text, "en" or "sv")`: text to the IPA espeak-ng gives the VITS voices (below) |
| `vits.olang` | text to speech: `Model.Load` (a transformers VitsModel), `Model.Piper` (a Piper voice's .onnx), `Symbols` (IPA to ids), `Speak` (ids to samples and every id's frames) |
| `tools/reference.py` | the reference the tests are held to: transformers, F32, on CPU |
| `tools/vits_reference.py` | VITS's references: transformers' VitsModel (vits-ljs) and onnxruntime (Piper's voices), with no noise |
| `tools/g2p_reference.py` | espeak-ng's phonemes (through phonemizer) for word lists and sentences, what `g2p.olang` is measured against |

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
on short commands, not the reference; `Text` keeps at least 3 s of silence after the sound, without which the model
never ends and repeats itself; `Options.NoRepeat` also stops greedy decoding at a phrase come three times running),
`--max-tokens`; with `--wake`: `--threshold`, `--pause`, `--longest`, `--quiet`.

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
- INT8 decoder (`--int8`, `Whisper.Load(..., q8 = true)`): the decoder's weights per-row INT8, its activations per-row dynamic INT8, 32-bit sums
  (the cross-attention's keys and values stay F32: a product of the encoder's rows); base hears Piper's English and Swedish as F32 does (speak test).
  On the 4-core x86 VM, base, 12.6 s clip, decoder per token: F32 20.2 ms (1 thread) / 10.9 (4); INT8 19.3 / 9.7. The gain there is small
  (compute-bound); the weights are a quarter of the bytes, which is what a Pi 4 decoder waits for - to measure there.

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

**Raspberry Pi 4, estimated** (not measured; the target): Cortex-A72 at 1.8 GHz, NEON (two 128-bit FMA pipes, no
BF16 or dot-product instructions) - 29 GFLOPS a core at peak against about 190 here, and a narrower core that reaches a
smaller share of it: the encoder, bound by arithmetic, about 8x slower; LPDDR4 at about 4-5 GB/s in practice against
about 25 here: the decoder, bound by reading its weights, about 5x slower at 4 threads and 6x at 1. F32 (`--bf16`
halves the memory and the decoder's reading, widened in software):

| model | threads | encoder (30 s window) | decoder / token | 24 s clip, ~75 tokens: RTF |
|---|---|---|---|---|
| tiny | 4 | ~2 s | ~42 ms | ~0.2 |
| tiny | 1 | ~4.7 s | ~85 ms | ~0.5 |
| base | 4 | ~4 s | ~90 ms | ~0.45 |
| base | 1 | ~10.5 s | ~160 ms | ~0.95 |
| small | 4 | ~13 s | ~200 ms | ~1.2 |

So on a Pi 4, tiny and base run faster than real time on 4 threads; a spoken command of 3 s through base costs about 5
s (the encoder's whole 30 s window, ten tokens), about 1.5 s with `--positions 300`, and through tiny about 2.5 s and 1
s. small is too slow for an assistant there. (A Pi 5 - Cortex-A76 at 2.4 GHz, about 10 GB/s - is about twice as fast
again.) All three models are multilingual: Swedish and English are detected (`LanguageCode()`) or fixed
(`--language sv`).

## Text to speech: the survey

Models with open code and permissively licensed weights, for a Pi (sizes are the published files):

| model | architecture | params / size | code | weights | input | fits a Pi 4 |
|---|---|---|---|---|---|---|
| kakao-enterprise/vits-ljs | VITS (end to end, HiFi-GAN decoder) | 36 M, 145 MB F32 (safetensors, transformers' `VitsModel`) | MIT | MIT; LJ Speech, public domain | IPA phonemes (espeak-ng's) | yes: VITS runs in real time on a Pi 4 (Piper) |
| rhasspy/piper-voices (e.g. en_US-ljspeech-medium) | VITS, Piper's training | 15-30 M, 60-75 MB a voice (ONNX) | MIT (rhasspy/piper; the successor piper1-gpl is GPL-3) | MIT repository; each voice's dataset licence in its MODEL_CARD (LJ Speech: public domain; some voices non-commercial) | espeak-ng phonemes | yes, the reference for Pi TTS |
| KittenML/kitten-tts-nano-0.1 | StyleTTS-like, ONNX | 15 M, 25 MB | Apache 2.0 | Apache 2.0 | espeak-ng phonemes | yes |
| hexgrad/Kokoro-82M | StyleTTS 2 + iSTFTNet | 82 M, 330 MB | Apache 2.0 | Apache 2.0 | misaki G2P (Apache) or espeak-ng | about real time on 4 cores; best quality here |
| myshell-ai/MeloTTS-English | VITS 2-like + BERT features | ~50 M + BERT | MIT | MIT | g2p_en (CMUdict) + BERT | heavier (the BERT) |
| rhasspy/piper-voices sv_SE-nst-medium | VITS, Piper's (trained by KBLab) | ~20 M, 63 MB (ONNX) | MIT | MIT repository; NST Swedish data **CC0** | espeak-ng phonemes (sv) | yes - the Swedish voice |
| rhasspy/piper-voices sv_SE-alma-medium, sv_SE-lisa-medium | VITS, Piper's | ~20 M, 63 MB | MIT | alma: CC BY 4.0; lisa: fine-tuned from a Norwegian voice, licence not stated | espeak-ng phonemes | alma yes (attribution); lisa unclear |
| facebook/mms-tts-eng, -swe | VITS | 36 M | MIT (transformers) | **CC BY-NC 4.0**: not permissive | characters | excluded |
| Coqui XTTS-v2 | GPT + HiFi-GAN | 470 M | MPL | **CPML, non-commercial** | text | excluded |

The best fit for an assistant in Swedish and English is Piper's VITS: sv_SE-nst-medium (CC0 data) and
en_US-ljspeech-medium (public-domain data), one architecture for both, real time on a Pi 4 as Piper is today, 22050 Hz.
kakao-enterprise/vits-ljs (MIT, the same architecture in safetensors and in transformers' `VitsModel`) is the way to
hold an implementation to a reference before reading Piper's ONNX files. Two pieces are the work, not the network (about
4 GFLOP a second of speech for Piper's medium voices): reading Piper's weights (`io/onnx.olang`; `vits.olang`, below) and a
grapheme-to-phoneme step - espeak-ng, which every candidate here was trained with, is GPL-3, so it would be our own,
matching espeak-ng's IPA for these voices: CMUdict (BSD-2) and rules for English, a lexicon and Swedish's fairly regular
spelling rules for Swedish: `g2p.olang`, below.

## Grapheme to phoneme

`G2p.Load()` (CMUdict, fetched once into `~/.cache/oann/g2p`), then `g.Phonemes(text, "en")` or `"sv"`: the IPA
string phonemizer gives with backend espeak, en-us or sv, `preserve_punctuation=True`, `with_stress=True` - words
separated by a space, the stress marks before the vowel, the punctuation phonemizer keeps kept where it was, numbers
spelled out as espeak reads them ("1999" is nineteen hundred ninety nine, "23" in Swedish ɕˌʉɡɔtrˈeː). No code or data
of espeak-ng is in it (GPL-3); espeak only writes the reference it is measured against.

**English.** A word is CMUdict's, or a suffix (-s, -ed, -ing, -ly, -ness ...), a prefix or a British spelling of a word
it has, or a compound of two; failing those, letter-to-sound rules (graphemes, long and short vowels by the consonants
after them, the stress by prefixes and endings). The ARPAbet is then written as espeak writes en-us: espeak's stresses
(a secondary next to the primary goes), the stressed vowels by ARPAbet and the unstressed ones mostly by the spelling
(each vowel phone aligned with its letters: "a-" ɐ, "-ed" after t or d ᵻd, "-ity" ᵻɾi, i ɪ), flapped t (ɾ), a glottal
before a last "-en" (ʔn̩), r-coloured vowels (oːɹ in "more", iə in "period"). Function words and the words of numbers
have espeak's forms, alone and in a phrase ("a" ˈeɪ alone, ɐ in a phrase; "in the" one word, ɪnðə), and a final ɚ
links to a vowel (ɚɹ).

**Swedish.** Spelling rules: k, g and sk soft before a front vowel at a stressed syllable's start (ɕ, j, ɧ - espeak
writes it sx), sj, stj, skj, tj, dj, gj, hj, lj, ng, nk, gn (ŋn), rs ʂ, rt t after a, å, ä, ö; a stressed vowel long
when one consonant follows (and before some r clusters, a stop and a liquid, av- and ut-), short before two, a double
one, ck, x or a last m; espeak's unstressed vowels (e and ä ə, o uː in an open syllable, i iː); the stress on the
first syllable, after be-, on -tion; a secondary on the second vowel after the primary in longer words, as espeak puts
it; and a table of 60 function words and the words of numbers. A pronunciation lexicon was surveyed: NST's Swedish
lexicon (Språkbanken, nb.no; 927 000 words in SAMPA) is CC0 and would serve, but it gives true pronunciations, and
espeak's sv rules often do not (sjösjuka is sxˈœsxʉkˌa, kort kˈɔt, bord bˈuːrd), so agreeing with espeak - what the
voice was trained on - is better served by rules aimed at its output; Wiktionary's pronunciations (CC BY-SA) were not
used.

**Agreement with espeak-ng 1.51 / phonemizer 3.6** (`python3 -I speech/tools/g2p_reference.py`, then `make test`; a file
`show` in `~/.cache/oann/ref/g2p` prints every line that differs). Word lists are counted from public-domain books
(Project Gutenberg): English from Pride and Prejudice, The Adventures of Sherlock Holmes, Tom Sawyer and Moby Dick,
Swedish from Lagerlöf's Bannlyst, Söderberg's Förvillelser and Gertrud Almqvist's I tolfte timmen. Exact: the share of
words (or sentences) the same as espeak's string; by edit distance: one less the edit distance over espeak's length, in
code points (phonemes, stress and length marks, spaces, punctuation):

| list | exact | by edit distance |
|---|---|---|
| English, the 2000 most frequent words | 90.4% | 97.8% |
| English, 200 frequent words CMUdict lacks (rules, suffixes, compounds) | 48.5% | 84.1% |
| English, 12 sentences (numbers, punctuation, abbreviations) | 8 of 12 | 99.3% |
| Swedish, the 1000 most frequent words (the rules were written looking at these) | 88.5% | 96.4% |
| Swedish, the next 1000 (held out) | 81.8% | 94.8% |
| Swedish, 12 sentences | 7 of 12 | 98.2% |

Most of what differs is espeak's own lexicon and rules: English stress in words espeak knows otherwise (address,
perfect), homographs (read, lead, wind), its FORCE and NORTH vowels (oːɹ, ɔːɹ) where the spelling does not tell them
apart, syncopes (every ˈɛvɹi, different dˈɪfɹənt); Swedish long vowels of inflected forms (köpte ɕˈøːptə), secondary
stresses and loanwords with a late stress (precis prəsˈiːs).

## Text to speech: VITS

`vits.olang` runs VITS (Kim, Kong and Son 2021) for inference, from phoneme ids to F32 samples at 22050 Hz, in two
publications of it: transformers' `VitsModel` (kakao-enterprise/vits-ljs: config.json, model.safetensors) and Piper's
voices (rhasspy/piper-voices en_US-ljspeech-medium and sv_SE-nst-medium: `<voice>.onnx`, `<voice>.onnx.json`). The
text is IPA as espeak-ng writes it (`g2p.olang` makes it); `Symbols` maps it to ids a code point at a time -
vits-ljs's vocab.json with a blank (0) around every symbol, as VitsTokenizer's `add_blank`; Piper's phoneme_id_map with
`^`, `_` after every phoneme and `$`, as Piper does.

    m := vits.Model.Piper(dir + "/sv_SE-nst-medium.onnx")!           # or vits.Model.Load(dir) for a VitsModel
    s := vits.Symbols.PiperMap(dir + "/sv_SE-nst-medium.onnx.json")!   # or vits.Symbols.Vocab(dir + "/vocab.json")
    o := vits.Options(m.C)            # NoiseScale, NoiseScaleDuration, LengthScale (the model's defaults), Seed
    wave, frames := m.Speak(s.Encode("hɛj vˈɛːɖɛn.")!, o)              # samples at m.C.Rate; each id's frames (256 samples)

**How it works.** The text encoder (the ids' embeddings times sqrt(192), six post-norm layers whose attention adds
relative-position keys and values for the offsets -4 .. 4, a feed-forward of two width-3 convolutions) gives every
id a prior's means and log-variances. The stochastic duration predictor runs in reverse: noise (times
NoiseScaleDuration) through its spline flows backwards - each a monotone rational-quadratic spline of 10 bins on -5 ..
5 whose widths, heights and slopes a dilated depth-separable convolution stack predicts from the text - gives every id's
log-duration, its frames ceil(exp(that) LengthScale). Each id's statistics are repeated over its frames and the latent
sampled (NoiseScale), then the residual coupling flow runs in reverse (four mean-only couplings of four WaveNet layers,
the channels flipped between them), and HiFi-GAN makes samples: transposed convolutions upsampling 256 times (8 8 2 2
for vits-ljs, 8 8 4 for Piper's medium) each followed by the mean of three residual blocks (ResBlock1 for vits-ljs,
ResBlock2 for Piper's), a tanh at the end. Noise comes from a seeded std/rand generator. Every signal is time-major,
so every convolution is one gemm of its patches gathered a run of rows at a time on each thread, the leaky ReLU before
each of HiFi-GAN's convolutions applied as the patches are gathered and a residual sum done by the product (beta 1);
a transposed convolution of stride s is s convolutions, one for each phase of its outputs.

**Piper's weights.** `io/onnx.olang` reads an ONNX file's initializers and its nodes' names (protobuf, no graph run).
Piper's names are original VITS's (`enc_p`, `dp`, `flow`, `dec`, `gamma`/`beta`, `convs_sep`, ...), mapped onto
transformers' one to one, with three twists: the export folded every weight-normed convolution's weight into an
unnamed constant (`onnx::Conv_8959`), which takes its name from the bias of the Conv node that reads it; the duration
predictor's affine flow keeps `exp(-log_scale)` as a constant or the input of an Exp; and the duration predictor's first
spline flow, which reverse never runs, is not in the file. Piper's medium network is the training config's (hidden 192,
HiFi-GAN from 256 channels, ResBlock2 with kernels 3 5 7 and dilations 1 2, 2 6, 3 12), which the .onnx.json does not
repeat; the low and high qualities are refused.

**Checks** (`make test`, where `~/.cache/oann` has the models and `ref/`, written by `python3 -I
speech/tools/vits_reference.py ~/.cache/oann/models ~/.cache/oann/ref`; skipped with a message otherwise). With no
noise (noise_scale and the duration predictor's 0), on three English sentences for vits-ljs and two for each Piper
voice (the ids `Symbols` gives their IPA are the reference's):

- every id's duration (frames) equal to transformers' and to onnxruntime's, so the waveforms have the same length;
- Piper's voices: the waveform within 2.9e-5 of onnxruntime's (en_US-ljspeech-medium, sv_SE-nst-medium);
- vits-ljs: within 1.4e-4 of transformers' F32 waveform and 1.7e-4 of transformers run in F64 - transformers' own F32
  waveform is up to 2.2e-4 from its F64 one (rounding through the flows and HiFi-GAN), so that is the floor in F32;
  the test holds both at 2.5e-4. (The latent before HiFi-GAN is within 1.5e-5.)

No ceiling of a duration landed on the other side of an integer in these cases; one could (exp(log-duration) within
F32 rounding of a whole number), and would then shift every later frame - the check would say so.

**Speed** (`make vitsbench ARGS="MODEL 3 1 4"`, bench/vits.olang: a sentence of 241-243 ids, 5.9-9.6 s of audio; the
4-core VM above, shared with another job while measured), seconds of compute a second of audio:

| model | threads | this VM | Raspberry Pi 4 (estimate) |
|---|---|---|---|
| Piper en_US-ljspeech-medium | 1 | 0.095 | ~0.76 |
| | 4 | 0.058 | ~0.47 |
| Piper sv_SE-nst-medium | 1 | 0.091 | ~0.73 |
| | 4 | 0.048 | ~0.38 |
| kakao-enterprise/vits-ljs | 1 | 0.69 | ~5.5 |
| | 4 | 0.30 | ~2.4 |

onnxruntime runs en_US-ljspeech-medium at 0.087 (1 thread) and 0.065 (4) on the same VM. The Pi 4 column is this VM's
times 8 - a Cortex-A72 (NEON, no BF16) is roughly 8 times slower on compute-bound work - an estimate, not a measurement:
Piper's medium voices speak faster than real time there on one core and about twice real time on four; vits-ljs (its
HiFi-GAN is 512 channels wide, about 13 times Piper's medium arithmetic) does not, which is why the voices are Piper's.

## Models, data and licences

Weights and clips are fetched by whoever runs the code; none are in this repository.

| what | used for | licence |
|---|---|---|
| openai/whisper-tiny, -base, -small (Hugging Face) | speech to text | Apache 2.0 on the Hub; OpenAI's release (github.com/openai/whisper, code and weights) MIT |
| jfk.wav (JFK's inaugural address, 1961, as whisper.cpp's sample) | test clip | public domain (a work of the US government) |
| LibriVox, Aesop's Fables vol. 1, sections 1-2 (archive.org `aesop_fables_volume_one_librivox`), cut to 24, 20 and 67 s | test clips | public domain (LibriVox) |
| the synthetic word of the kws test | wake-word test | made by the test |
| transformers, torch (reference.py only) | the reference | Apache 2.0, BSD-3; not vendored, not needed to run |
| CMUdict (cmusphinx/cmudict, `cmudict.dict`, fetched at run time) | English pronunciations (g2p) | BSD-2 |
| espeak-ng 1.51, phonemizer (g2p_reference.py only) | the g2p reference | GPL-3; run offline to measure, not vendored or linked, not needed to run |
| Project Gutenberg books 1342, 1661, 74, 2701, 39147, 21862, 77571 (fetched by g2p_reference.py) | g2p word lists | public domain (in the US) |
| NST Swedish pronunciation lexicon (Språkbanken, nb.no) | surveyed for g2p, not used | CC0 |
| kakao-enterprise/vits-ljs (Hugging Face: config.json, vocab.json, model.safetensors) | text to speech, the transformers-checked VITS | MIT; trained on LJ Speech, public domain |
| rhasspy/piper-voices en_US-ljspeech-medium (`.onnx`, `.onnx.json`) | English voice | MIT (the repository); dataset LJ Speech, public domain (its MODEL_CARD) |
| rhasspy/piper-voices sv_SE-nst-medium (`.onnx`, `.onnx.json`) | Swedish voice (trained by KBLab) | MIT (the repository); dataset NST Swedish (Språkbanken, nb.no), CC0 (its MODEL_CARD) |
| onnx, onnxruntime (vits_reference.py only) | Piper's reference | Apache 2.0, MIT; not vendored, not needed to run |
| Wiktionary pronunciations | not used | CC BY-SA |

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
