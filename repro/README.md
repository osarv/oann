# repro

Minimal programs for olang issues found while writing oann. Each file's header says what it shows and what was expected.
Last re-checked against olang 3b47ad852 (2026-11-21): of the bugs, only `tempref` still reproduces; `fieldname` is a
deliberate rule. The fixed files are kept as regression cases.

| file | status | why |
|---|---|---|
| `fieldname.olang` | reproduces | D3a: a constructor field may not share a function's name in its module. A deliberate rule, kept as a record of the friction |
| `joinparen.olang` | fixed (message) | E13b's rule stays deliberate; the message is E11b again ("'(' after a piece of text calls it"), naming the way out |
| `lambdacapture.olang` | fixed | builds and prints 0 twice (2026-11-21) |
| `nullmigrate.olang` | fixed | `olang fix` no longer crashes: it leaves the T2c it cannot place to be fixed by hand and names it (2026-11-21) |
| `fieldofresult.olang` | fixed | builds and prints ok: region inference (QCB) replaced the O10d refusal. Kept as a regression case |
| `membercascade.olang` | fixed | one error, `S has no member 'flag'`, no second one |
| `nestedtry.olang` | no longer a limitation | a `try` takes the whole expression after it (E24): `try entry["shape"].AsArr() catch { ... }` covers the lookup and the conversion with one clause. The old nested form still gets R9, as the spec says |
| `variadicat.olang` | fixed | `x[1, 1]` calls a variadic At with two indices; its test passes (2026-11-21) |
| `traittype.olang` | fixed | T2 "unknown type 'Tensor'" at the trait's signature (2026-11-21) |
| `colfields.olang` | fixed | `Columns<C>.Fields` holds lower-case fields in C's own module: the generic clears them, prints 0 (2026-11-21) |
| `genref.olang` | fixed | the generator builds and prints 1 (olang HoldingGenerator) |
| `tempref.olang` | reproduces (new) | a temporary passed to a `&` parameter of a function whose result copies it into a struct also holding a reference, that result itself a temporary passed by reference, is built in the referent's scope at every call: resident memory grows without bound. ad/graph took tensors by value to avoid it |
| `allocfacts.olang` | fixed | both asserts hold on olang master of 2026-11-15 (G42 counts running ways only, a fixed array in place is no allocation). Kept as a regression case |
| `methodfacts.olang` | fixed | `Box<I64>.Fill.Allocates` is read (ad/graph pins `Graph<F32>.Forward.Allocates` directly). Kept as a regression case |
| `tempfacts.olang` | fixed | both asserts hold (2026-11-21); the locals bound for it in ad/, kernels/ and dyn/ may go |
| `reassignfacts.olang` | fixed | both asserts hold (2026-11-21) |
| `tryindex.olang` | fixed | builds and runs: `try` no longer makes an array index inside its operand a checked one (indexing is unchecked, `try a.Get(i)` the checked form) |
| `learnerarg.olang` | fixed | builds and prints nothing (2026-11-21) |
| `factcascade.olang` | fixed | with an unused import added, the M5a error alone (2026-11-21) |
| `crossinline.olang` | fixed | a program past ThinFrom (8 MB of IR) links by ThinLTO, which neither inlines a body over ImportLimit instructions from another object nor vectorizes again: gemm 16 x 16 x 11 in F64 is 2.2x slower in a program that also imports ad/graph, and elem's activations called fastExp64 per element (6x slower) until they got array-wide forms. Found by bench/smallgraph |
| `joinleak.olang` | fixed | 200 000 joins of two tasks grow the resident set by 92 pages (about 2 bytes a join; 110 before) |
| `spawnleak.olang` | fixed | 100 000 joins grow it by 7 pages |
| `condnull.olang` | fixed | builds and prints 3 |
| `vnnitile.olang` | fixed | olang's grouped lane total (E39's `Sum<A>(n)`) builds the VNNI row to vpdpbusd; kernels/gemm's INT8 dots use it |


The `Status` line in each file's header repeats the row.
