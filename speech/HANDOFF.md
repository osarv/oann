# speech: hand-off (2026-11-23, stopped by the usage guard)

Read first: this note, then speech/README.md (API, checks, speed, licences). Last commits: 5fcd249 (g2p clock
times), fb13d23 (speak.olang), 91bc1bd (shortened windows keep 3 s silence, Options.NoRepeat). Nothing is in progress;
the tree was clean. Users: the OAgent home session (session_01DLX3U7kkfPnv219zp4wSrU) runs Whisper, kws.Detector and
speak.Voice from oann; e75-speech (session_01DLX3U8EBgCK5YZpKCPmx4N) reuses mel/kws.

aarch64 under qemu: mel, whisper and transcribe test files pass with tiny (logits 7.6e-5); kws and g2p tests pass under qemu (2026-11-24; reference-file parts skipped, 12 min); speak/vits not yet run.

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

ARM toolchain in a fresh Ubuntu 24.04 container: apt install clang-19 libc6-dev-arm64-cross libc6-arm64-cross libgcc-13-dev-arm64-cross binutils-aarch64-linux-gnu qemu-user-static lld-19; ln -s /usr/lib/gcc-cross/aarch64-linux-gnu/13 /usr/aarch64-linux-gnu/lib/gcc/aarch64-linux-gnu/13; PATH with a dir whose `ld` is ld.lld-19; OLANG_SYSROOT=a dir with usr/aarch64-linux-gnu, usr/include, usr/lib, lib symlinked to /usr/aarch64-linux-gnu[/...].
