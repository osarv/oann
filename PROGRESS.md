# oann catch-up to olang v1 (374c1e3c) - hand-off note

Round 1: get every oann file compiling with today's olang and `make test` green, behaviour kept, rewriting only what the
language forces. The compiler: `/home/user/wt/olangv1/build/olang` (olang master 374c1e3c, built with `make`).

## How a file is migrated (what worked)

1. `import "std/linalg"` -> `import "linalg"` (or `"../linalg"` from a subfolder): std has no linalg since olang
   b6e4a103; `linalg.olang` here is its last version, brought to today (committed a7276a3, 35 tests pass).
2. Protect matrix deep copies before `olang fix`: `sed -i 's/\.Clone()/.ClonePH()/g'`, and back after. Matrix keeps
   `Clone()` (a matrix is a view holding its data by reference, so the supplied `Copy()` would share the data);
   clone_migrate would otherwise turn every `.Clone()` into the shallow `.Copy()`.
3. Matrix's copy-from method is `m.CopyFrom(src)` (was `m.Copy(src)`, which now clashes with the supplied Copy):
   `sed -i 's/\.Copy(\([^)]\)/.CopyFrom(\1/g'`.
4. oann is written before QY/G8b: run `tools_bare_typevars` and `tools_syntax_migrate --old` (built by `olang fix` into
   `~/.cache/olang/fix/<key>/`) before `olang fix FILES`; then `tools_precedence_migrate` (the code predates E5).
5. Then compile and fix by hand in dependency order.

## Ownership this round (coordinator, 2026-11-15)

- wt-oannport (this note's author): kernels, conv, ops, quant, nn, layers, optim, vision, checkpoint, train, generate,
  the makefile, README.md, PROGRESS.md.
- Another agent: sparse, store, tokenizer, datasets/*, bench/*.py and *.sh ports, repro/ re-checks.
- Later agents: circuit, consolidator, agent, board, examples/, bench/*.olang.

## Done

- linalg.olang (a7276a3).

## Next

- kernels, then nn and optim, pushed as each lands.
