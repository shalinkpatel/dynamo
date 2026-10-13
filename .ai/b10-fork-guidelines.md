# Baseten Fork Guidelines

This repository is Baseten's fork of [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo).
`main-v1.2.0`, the default branch, was cut from upstream commit `bed9f2693` (about
2026-06-01) and carries Baseten patches on top. Every Baseten line inside an upstream file
is a potential conflict when the fork moves to a newer upstream release, so these rules
aim to keep fork-only code small, isolated and easy to drop.

## Check upstream before writing fork-only code

The fork lags upstream by months, so the problem may already be solved there. Before
designing a fork-only change, search upstream's merged and open PRs and issues, and its
current `main`:

```bash
gh search prs --repo ai-dynamo/dynamo <keywords>
gh search issues --repo ai-dynamo/dynamo <keywords>
```

If upstream has it, backport it rather than reimplementing it.

## Backporting an upstream change

- Backport it in its own PR, separate from the feature that needs it, titled like
  `feat(frontend): backport SSE keep-alive (upstream #12453)`.
- Keep upstream's names, signatures and semantics verbatim, so the next upgrade merges
  cleanly. Note any adaptation the fork's older structure forced.
- Take only what the fork needs. Say in the PR which parts of the upstream PR were left
  out.
- Record it in `baseten-changelog.md` with status `upstream-sync`, the upstream PR, and
  "drop when the fork advances to an upstream release containing it".

## Keep Baseten code out of upstream files

- Put Baseten-only code in `b10_`-prefixed files and modules, for example
  `lib/llm/src/kv_router/b10_bid.rs`. Leave only minimal hooks (a field, a call, a
  `mod` line) in upstream files.
- When the code needs an upstream type's private fields, make it a child module of the
  upstream module (for example `scheduling/filter/b10_draining.rs`); a child module can
  read its parent's private items.
- Group fork-only fields in a fork-owned struct (one field in the upstream type) instead
  of adding them one by one.
- Prefix Baseten-only tests and test helpers with `b10_`.

## Changelog entries

`baseten-changelog.md` is the patch ledger for future upgrades; read its guidelines at the
top before editing it.

- Keep an entry to one or two lines that tell someone replaying the patch what it does,
  with a link to the PR. Don't repeat the PR description.
- Add a new top-level `PATCH-NNN` section only for a substantial standalone change
  (around 500 lines or more) or a distinct future rebase decision; otherwise extend the
  closest existing section.
