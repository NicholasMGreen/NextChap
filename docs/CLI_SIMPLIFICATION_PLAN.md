# CLI Simplification Plan

Status: implemented (2026-11-02). Owner decisions recorded in the "Decisions"
section below. Implementation notes at the end.

## Goal

Make the CLI require fewer flags, guess intent instead of demanding exact
spellings, and surface what you can do at every moment — while keeping the
generation pipeline, journaling, and all core invariants untouched.

## Decisions (owner-confirmed)

1. Bare `nextchap` → **interactive menu** (book picker → action picker).
2. Book targeting → **fuzzy matching everywhere**; no persistent "active
   book" state.
3. Flags → **smart defaults**, recurring personal choices move to
   `nextchap.toml [defaults]`.
4. Command set → **consolidated** into a small verb set (see Phase 3).
5. Tab-complete → install zsh completion (`nextchap --install-completion`);
   Ghostty is the terminal, zsh the shell.

## Guardrails (unchanged invariants)

- Pipeline core untouched: `engine.py` journaling, `sync()` idempotency,
  atomic writes, `Reporter` protocol, book-ID validation.
- All four checks (pytest, ruff check, ruff format, mypy --strict) green
  after every phase.
- Tests never call the network (`FakeLLM`, hash embeddings).
- No new runtime dependencies in phases 1–5 (stdlib `difflib` for fuzzy,
  rich `Prompt` for pickers).

---

## Phase 1 — Fuzzy book resolution (foundation)

New module `src/nextchap/cli/resolve.py`:

```python
def resolve_book(settings, partial: str | None) -> BookPaths  # never raises "not found" cold
```

Resolution ladder (falls through on miss):

1. **Omitted** (`nextchap generate -n 3`): 1 book → auto-select with
   confirmation line `using book: <id>`; 2+ books → numbered picker.
2. **Exact ID match** → use it.
3. **Prefix match** (`deadp`): unique → use it; ambiguous → picker
   pre-filtered by prefix.
4. **Fuzzy fallback** (`difflib.get_close_matches`) catches typos.
5. **No match** → error panel listing available books + menu hint.

`BOOK_ID` becomes optional in every book-taking command. Shell tab-completion
(existing, `nextchap --install-completion`) handles full IDs at the shell;
the runtime fuzzy layer covers partial/ambiguous input. `doctor` gains a
check: completion installed?

Tests: resolution ladder unit tests (0/1/2+ books, unique prefix, ambiguous
prefix, typo, no match).

## Phase 2 — Defaults & config-driven flags

Extend `config.py` with `DefaultsConfig`:

```toml
# nextchap.toml (project) or user config.toml
[defaults]
chapters = 3        # replaces need for -n
max_cost = 1.0      # always-on budget guard
review = false      # was --review
preview = true      # was --no-preview
critic = true       # was --no-critic
export_epub = true  # was --no-epub
```

Precedence: CLI flag > `nextchap.toml [defaults]` > built-in default
(reuses the existing pydantic-settings source chain). Flags remain as
overrides; the point is not needing them. Import's `--yes` becomes default
in non-TTY contexts; the menu flow confirms interactively.

Tests: precedence matrix (flag beats toml beats builtin).

## Phase 3 — Command consolidation

Target surface (visible commands):

| New                          | Absorbs                        | Notes                                            |
| ---------------------------- | ------------------------------ | ------------------------------------------------ |
| `nextchap import FILE`       | import                         | + fuzzy/auto book-id                             |
| `nextchap generate [-n N]`   | generate                       | defaults from Phase 2                            |
| `nextchap status [BOOK]`     | books list/info, state show    | chapters, world state, spend, index, journal     |
| `nextchap fix [BOOK]`        | state rebuild/rollback, index rebuild, clean | diagnoses + numbered repair choices |
| `nextchap export [BOOK]`     | export epub                    | unchanged                                        |
| `nextchap config/models/doctor` | unchanged                   | already fine                                     |

Old names (`books`, `state`, `index`, `clean`) remain as hidden aliases
printing a one-line deprecation pointer; removed in a later release.
Unknown-command typos get "did you mean…?" (extends the existing
UsageError handling in `main()`).

## Phase 4 — Interactive menu (`nextchap` with no args)

Zero new dependencies: rich numbered pickers (arrow-key `questionary` is an
optional later upgrade).

```
NextChap — 3 books · $2.41 total spend
 1) deadpool_twilight_v2   17 ch · 4 generated · mid-arc
 2) chaos_derry            82 ch
 3) lord_unknown           54 ch
Book? [1-3]: 1
 1) generate next chapter(s)
 2) status / continuity report
 3) fix (rebuild index, roll back state…)
 4) export epub
 5) switch book
Action? [1-5]:
```

The menu is a shell around the verb commands — no new business logic.
Reuses the Phase 1 resolver. `--plain` and non-TTY runs keep today's
behavior (no forced interactivity).

## Phase 5 — Informativeness pass

- `Examples:` epilog (2–3 copy-paste lines) on every command help.
- `error_panel` gains contextual hints ("no books yet → nextchap import …").
- README quickstart rewritten for the new surface; AGENTS.md command table
  updated.

## Phase 6 — Tests, docs, sign-off

- End-to-end CLI tests: menu via piped input, fuzzy resolution, status/fix.
- All four checks green; BUGS.md cross-referenced.

## File-touch estimate

- New: `cli/resolve.py`, `cli/menu.py`
- Edits: `config.py`, `cli/app.py`, `cli/commands_extra.py`, `cli/ui.py`
  (help epilogs), README, AGENTS.md
- New tests: ~10

Roughly three working sessions; Phases 1 + 5 alone are a large UX win if a
first cut is wanted sooner.

## Implementation notes

- `cli/resolve.py` implements the exact → prefix → fuzzy ladder and the Rich
  picker. Validation still happens before any path is built, so traversal-like
  IDs keep raising `InvalidBookId`.
- `status` without a book lists all books when there are zero or several;
  `status BOOK` shows the detailed report. This is the one intentional
deviation from the "always pick a book" ladder, so `books list` has a visible
  replacement.
- `export` accepts both `nextchap export BOOK` and the old
  `nextchap export epub BOOK` form (the latter prints a deprecation line).
- `fix` diagnoses summaries, world state, journal, master-book noise and index
  metadata. It offers numbered repairs interactively and
  `--rebuild-index` / `--rebuild-state` / `--clean` for scripts.
- `config show` now strips nested `None` values before TOML output (needed by
  the optional `[defaults]` booleans).
- New tests: `tests/test_resolve.py` plus end-to-end status/fix/export/menu and
  defaults-precedence cases in `tests/test_cli.py`.
