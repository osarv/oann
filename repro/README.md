# repro

Minimal programs for olang issues found while writing oann. Each file's header says what it shows and what was expected.
Last re-checked against olang f73ca3c4f (2026-11-21). A repro is removed once olang fixes it (git keeps them:
`git log --diff-filter=D -- repro/`); what is left still reproduces, or records a deliberate rule.

| file | status | why |
|---|---|---|
| `fieldname.olang` | reproduces | D3a: a constructor field may not share a function's name in its module. A deliberate rule, kept as a record of the friction |
| `tempref.olang` | reproduces (new) | a temporary passed to a `&` parameter of a function whose result copies it into a struct also holding a reference, that result itself a temporary passed by reference, is built in the referent's scope at every call: resident memory grows without bound. ad/graph took tensors by value to avoid it |


The `Status` line in each file's header repeats the row.
