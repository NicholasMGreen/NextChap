# Plan: `nextchap query` — free-text search over a book's index

Status: implemented on `feature/query-command` (2026-11-02).
Depends on: Phase 1 of `CLI_SIMPLIFICATION_PLAN.md` (fuzzy book resolver).

## Decisions (owner-confirmed)

1. **Passages only, no LLM.** A query returns retrieved chunks with chapter
   references. Zero API cost, zero risk of invented answers.
2. **One book only.** No cross-book search. The book is resolved the same
   fuzzy way as every other command.
3. **Always one-shot.** One query → results → exit. No REPL loop, not even
   in the menu.

Owner requirements (fixed):
- Must be reachable as `nextchap query` on the CLI.
- Must appear in the interactive menu's action list (alongside generate,
  fix, export, …) after picking a book.

## Feature spec

### CLI

```
nextchap query [BOOK] "question text"     # passages matching the question
nextchap query [BOOK] --json              # machine-readable results
nextchap query [BOOK] -k 10               # more/fewer results (default: retrieval.top_k = 6)
```

- `BOOK` optional: exact ID → unique prefix → fuzzy (`difflib`) → picker,
  exactly like other commands (Phase 1 resolver).
- No `--max-cost`, no model selection: there is no LLM call at all.

### Output (human mode)

```
Query: what does Walter know about Wade's powers
Book: deadpool_twilight_v2 · 6 results (hybrid search)

 1) Ch 09 · (0.83)  …Walter watched from the porch, the way he always did
                     when Wade thought no one was looking…
 2) Ch 03 · (0.77)  …
```

- Rank, chapter number, score, excerpt (a few wrapped lines, the best
  matching sentences shown, `…`-trimmed).
- Note printed once when the collection metadata's embedding model differs
  from the configured one → point at `nextchap fix` (rebuild index).

### Menu integration

Add a `query` action to the picker menu (Phase 4 of the CLI plan; until that
phase lands, the action list is just extended in place). Flow:
book picked → `Question?` prompt (rich `Prompt`, empty = back to menu) →
one-shot results → back to menu. One question per entry, per decision 3.

## Implementation sketch

- New: `cli/query.py` (command + result rendering) — or a `query_cmd` in
  `cli/app.py` if it stays small.
- Reuses the existing retrieval stack unchanged:
  - query embedding via the configured backend (`retrieval.embed_backend`,
    local Qwen by default — already installed);
  - `VectorStore` hybrid search (dense + BM25 → RRF, weights from config);
  - `retrieval.top_k` / `candidates` as defaults, `-k` overrides.
- Read-only: opens the book, queries the store, prints. No locks, no
  journal, no writes, no model calls. Chroma client opened via the normal
  `VectorStore` path (never touch the collection folder directly).

## Edge cases

| Case | Behavior |
|---|---|
| Book has no index / empty | Error panel: "run `nextchap fix` to build the index" |
| Embedding-model mismatch (`IndexMismatch`) | Error panel: "run `nextchap fix` (rebuild)" — never auto-rebuild on a read path |
| Query with no arguments | In TTY: prompt for the question; non-TTY: usage error |
| No results | "No passages matched — try fewer words or another phrasing" |

## Tests (no network, hash embeddings)

1. E2E: import fixture book → `query "…"` returns the chapter that contains
   the queried phrase, with the right chapter number.
2. Resolver integration: prefix + typo + omitted book (after Phase 1).
3. `--json` shape: list of `{chapter, score, excerpt}`.
4. `IndexMismatch` → clean error with fix hint (no rebuild attempted).
5. Menu action present in the picker list.

## Non-goals

- No LLM synthesis mode (can be added later behind `--answer`; deliberately
  deferred).
- No cross-book search.
- No saved search history.
