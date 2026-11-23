# speech: hand-off (2026-11-23)

Done and pushed: Whisper (mel, whisper, transcribe, listen/oann-listen), kws (wake word), bench/speech, README (licences,
VM speed, Pi 4 estimate, TTS survey). Exact against transformers 5.24 for tiny/base/small (see README, Checks).

Also done: speech/g2p.olang (en, sv; agreement with espeak-ng in README), speech/vits.olang + io/onnx.olang (Piper
voices exact to onnxruntime, vits-ljs to transformers), speech/speak.olang (Voice.Load(lang), Speak(text)); its test
speaks English and Swedish and hears them back through Whisper base.

Next: measure on a real Pi 4; a trained wake word (data to record or synthesise - Speak can make positives in both
languages, kws.Train); INT8 weights for the Pi 4 decoder (memory-bound there); the G2P's English words outside CMUdict
(about half differ from espeak).

Environment notes: tests need `OLANG_AUTORACE=0` (clang-19's TSan makes the race build very slow); the ARM build needs
clang-19 + libc6-dev-arm64-cross + qemu-user-static (`QEMU_LD_PREFIX=/usr/aarch64-linux-gnu`); `OANN_SPEECH_QUICK=1`
checks tiny only. Reference data: ~/.cache/oann/{models,audio,ref}, made by speech/tools/reference.py.
