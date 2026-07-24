# .claude/ — committed Claude Code configuration

Everything here **except** `settings.local.json` is committed and travels with
the repo.

| File | Purpose | Committed? |
| ---- | ------- | ---------- |
| `settings.json` | shared settings, incl. `outputStyle: Learning` | yes |
| `settings.local.json` | your machine-only overrides | no (git-ignored) |
| `MEMORY.md` | project memory index, loaded each session | yes |
| `memory/*.md` | project memory bodies (one fact per file) | yes |
| `link-memory.ps1` | bridges memory into the harness path | yes |

## Why memory needs a bridge

Project memory is the source of truth **here in the repo**, so it is committed
and shared. But the Claude Code harness reads memory from a fixed, machine-local
path — `~/.claude/projects/<repo-slug>/` — that cannot be reconfigured to point
at the repo.

`link-memory.ps1` reconciles the two: it makes the harness's `memory/` a
directory **junction** into `.claude/memory/`, and its `MEMORY.md` a **hard
link** to `.claude/MEMORY.md`. After that, anything Claude writes to memory lands
in the committed repo files, and `git commit` captures it.

Junctions and hard links are local filesystem objects; git does not track them.
So the *content* travels with a clone, but the *link* does not.

## First-time setup on a new machine

```powershell
git clone https://github.com/jackwiencek/decoder-only-transformer.git
cd decoder-only-transformer
pwsh -File .claude/link-memory.ps1     # one-time; re-establishes the bridge
```

The script is safe to re-run and backs up any pre-existing real memory files
before linking.
