# repro

Minimal programs for olang issues found while writing oann. Each file's header says what it shows and what was expected.
Last re-checked against olang 374c1e3c (2026-11-15); the files were brought to that syntax (`olang fix`, then by hand
where the library changed) so each builds or fails as its header says.

| file | status | why |
|---|---|---|
| `fieldname.olang` | reproduces | D3a: a constructor field may not share a function's name in its module. A deliberate rule, kept as a record of the friction |
| `joinparen.olang` | reproduces | E13b: `$alpha (" x")` calls what was rendered. The rule is deliberate, but the message is now `E13b: F64 is not a function`, not the E11b that named the way out (`$(...)`) - worth restoring |
| `nullmigrate.olang` | reproduces (new) | `olang fix` crashes in `tools/null_migrate.olang:162` (slice bounds out of range) on a T2c error reported inside a generic's instance; the tools after it do not run |
| `fieldofresult.olang` | fixed | builds and prints ok: region inference (QCB) replaced the O10d refusal. Kept as a regression case |
| `lambdalend.olang` | fixed | builds and prints ok: O17 no longer refuses the lent value. `linalg.View` now fails with `ShapeError`, so the matrix is made with the `Matrix` constructor |
| `membercascade.olang` | fixed | one error, `S has no member 'flag'`, no second one |
| `nestedtry.olang` | no longer a limitation | a `try` takes the whole expression after it (E24): `try entry["shape"].AsArr() catch { ... }` covers the lookup and the conversion with one clause. The old nested form still gets R9, as the spec says |
| `resultgrowth.olang` | fixed | resident memory stays flat over 4 000 000 calls (1720 kB before, 1792 kB after); O26a no longer moves the unrelated local into the result scope |
| `variadicat.olang` | reproduces (new) | E31: a variadic `At(idx ...I64)` is the operator for one index only; `x[i, j]` is an error. An N-d tensor (rank known at run time) cannot declare `t[i, j]` for every rank |
| `tryindex.olang` | fixed | builds and runs: `try` no longer makes an array index inside its operand a checked one (indexing is unchecked, `try a.Get(i)` the checked form) |

The `Status` line in each file's header repeats the row.
