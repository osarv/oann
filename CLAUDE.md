# oann: working rules

- **Parallel sessions** (the user, 2026-11-17): several Claude sessions (the local machine, cloud VMs) may work on olang, oann and oagent at the same time. Commits by another session are the user's own work; any file in any of the repos may be changed as the work needs. Pull before starting, rebase or merge from origin before every push, and never overwrite or force-push over another session's pushed work.
- **Commits** are authored as Oscar Arvidsson <oscar.arvidsson@protonmail.com>, with no Claude attribution lines.
  Push always; never delete branches.
- The olang project's memory (`../olang/.claude/memory/`) holds the wider plan, the work order and the standing rules;
  the design is `docs/redesign.md`, progress `PROGRESS.md`.
