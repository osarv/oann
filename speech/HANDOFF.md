# speech: hand-off (2026-11-23)

Done and pushed: Whisper (mel, whisper, transcribe, listen/oann-listen), kws (wake word), bench/speech, README (licences,
VM speed, Pi 4 estimate, TTS survey). Exact against transformers 5.24 for tiny/base/small (see README, Checks).

In flight (two agents, started 22:55): speech/vits.olang (VITS on kakao-enterprise/vits-ljs, held to transformers'
VitsModel with zero noise) and speech/g2p.olang (English via CMUdict + rules, Swedish via rules/lexicon, measured
against espeak-ng/phonemizer offline; espeak-ng is GPL and is never vendored). If they stopped: check `git log` for
their commits and pick up from there.

Next after those: run Piper's sv_SE-nst-medium (CC0 data) - its weights are ONNX, so either an ONNX reader or a one-off
conversion to safetensors; a Speak(text, lang) API for the OAgent home session; a trained wake word (data to record or
synthesise, kws.Train); INT8 weights for the Pi 4 decoder (memory-bound there).

Environment notes: tests need `OLANG_AUTORACE=0` (clang-19's TSan makes the race build very slow); the ARM build needs
clang-19 + libc6-dev-arm64-cross + qemu-user-static (`QEMU_LD_PREFIX=/usr/aarch64-linux-gnu`); `OANN_SPEECH_QUICK=1`
checks tiny only. Reference data: ~/.cache/oann/{models,audio,ref}, made by speech/tools/reference.py.
