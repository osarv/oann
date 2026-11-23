# repro

Minimal programs for olang issues found while writing oann. Each file's header says what it shows and what was expected.
Last re-checked against olang f73ca3c4f (2026-11-21). A repro is removed once olang fixes it (git keeps them:
`git log --diff-filter=D -- repro/`); what is left still reproduces, or records a deliberate rule.

| file | status | why |
|---|---|---|
| `fieldname.olang` | reproduces | D3a: a constructor field may not share a function's name in its module. A deliberate rule, kept as a record of the friction |
| `unknowntype.olang` | fixed (olang 5525e1d95: error[M9] "module std/time declares no 'Instant'") | a parameter typed by a module member that does not exist (`time.Instant`) passes the checker and stops the build in lowering ("does not lower an unknown type") instead of a check error naming it |


The `Status` line in each file's header repeats the row.
