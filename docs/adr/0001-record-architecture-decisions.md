# ADR 0001: Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-07-22

## Context

This project is a learning exercise whose value is partly the code and partly
the ability to explain *why* the code looks the way it does. Small ML projects
accumulate dozens of quiet decisions — vocabulary size, context length, weight
tying, optimizer betas — that look arbitrary six weeks later and are impossible
to defend in an interview.

Git history records *what* changed. It records *why* only accidentally.

## Decision

Every decision with a defensible alternative gets a numbered ADR in
`docs/adr/NNNN-short-slug.md`, using this format: Context, Decision,
Consequences, and where useful Alternatives considered.

ADRs are immutable once accepted. If a decision is reversed, write a new ADR
that supersedes the old one and mark the old one `Superseded by NNNN`.

## Consequences

- Slight overhead per decision, paid back the first time someone asks
  "why 8192 tokens?"
- The `docs/adr/` directory doubles as interview prep.
- Requires discipline: a decision made in conversation and never written down is
  a decision that will be re-litigated.
