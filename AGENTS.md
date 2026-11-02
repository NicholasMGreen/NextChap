# AGENTS.md — NextChap

Guidance for AI coding agents (and humans) working in this repository.

## What this project is

NextChap is a Python CLI (`nextchap`) that continues a web novel. It imports an `.epub`, `.txt` or master-book `.md` file and builds long-term story memory. Then it plans, writes, checks and commits new chapters through OpenRouter. It must stay a CLI: no web UI or server.

The model setup is budget-only. Defaults live in `src/nextchap/config.py` (`ModelsConfig`):

| Role | Model | Reasoning |
|---|---|---|
| writer | `anthropic/claude-haiku-5.5` | `none` |
| critic | `anthropic/claude-haiku-5.5` | `none` |
| planner | `deepseek/deepseek-v4.1-flash-1017` | `low` |
| analyst | `deepseek/deepseek-v4.1-flash-1017` | `none` |

Each model is the other's fallback. Don't hard-code model IDs anywhere else.

## Commands

```bash
uv sync                                  # install + dev tools (.venv)
uv sync --extra local-embed              # + torch/sentence-transformers for local embeddings
uv run nextchap --help
uv run pytest                            # ~90 tests, no network, ~5 s
uv run ruff check . && uv run ruff format --check . && uv run mypy   # mypy is --strict
```

All four checks must pass before you finish a change. Tests never call the network. They use `tests/fakes.py::FakeLLM` and the `hash` embedding backend (`NEXTCHAP_RETRIEVAL__EMBED_BACKEND=hash`).

## Layout

```
src/nextchap/
  config.py        pydantic-settings: flags > NEXTCHAP_* env > .env > ./nextchap.toml > user config
  models.py        OpenRouter /models registry (cached 24h) + offline fallback prices
  errors.py        NextChapError subclasses; each carries a user-facing `hint`
  llm/             client.py (httpx → OpenRouter: SSE streaming, retries, continuation,
                   structured outputs + repair, cost), base.py (LLM protocol), cost.py, schema.py
  ingest/          text.py (chapter detection), epub.py (read/write), cleanup.py, master_book.py
  memory/          world_state.py (models, delta apply, v0 migration, snapshots), summaries.py
  retrieval/       store.py (Chroma + BM25/RRF), embed.py, bm25.py, rerank.py, context.py (budgets)
  pipeline/        engine.py (sync + journaled generation), planner/writer/critic.py, plan.py,
                   journal.py, book.py (per-book files), manage.py (no-LLM ops), estimate.py
  prompts/         *.md templates (string.Template `$var`); bump PROMPT_VERSION when changing
  storage/         paths.py (book-ID validation, layout), atomic.py, lock.py
  cli/             app.py (Typer commands, main()), resolve.py (fuzzy book IDs + picker),
                   menu.py (bare `nextchap` book → action menu), commands_extra.py,
                   ui.py (Rich), context.py
tests/             unit + engine + CLI end-to-end; fakes.py, conftest.py
docs/MODERNIZATION_PLAN.md            design rationale
```

## Core invariants (don't break these)

- **The source of truth for chapter text is `master_book.md`** (`## Chapter N: Title` blocks). `Engine.sync()` reconciles everything else against it on every run:
  - summaries are compared by content hash;
  - chapters newer than `world_state.chapter_count` get world-state deltas;
  - stale arcs are re-summarized;
  - the index is compared by hash.
  Sync must stay idempotent: a second `sync()` makes zero model calls.
- **Journaled commits.** `Engine._generate_one` advances `journal.json` through `started → planned → written → reviewed → analyzed → committing → committed`. All artifacts are staged in `staging/chNNNNN/`. `_commit` is idempotent and writes in this order: master book, world state, index (`upsert` with deterministic chunk IDs), summaries, `book.json`. Never apply a world-state delta inside commit; commit writes the *staged* state.
- **Book IDs are validated** (`storage/paths.py::validate_book_id`) before touching disk. Never build a path from an unvalidated ID.
- **Every file write goes through `storage/atomic.py`** and uses UTF-8.
- **World-state updates** go through `apply_delta`, which returns a new object, never mutates its input, and raises `DeltaRejected` on bad data. Threads and characters are referenced by stable IDs (`t001`, `char_lin_feng`).
- **The LLM client retries only `RetryableLLMError`** (429, 5xx, timeouts, empty answers). Auth and credit errors are `FatalLLMError` with hints. It checks `meter.check_budget()` before every request and only sends parameters listed in the model's `supported_parameters`. Haiku 5.5 rejects `temperature` and `top_p`.
- **The embedding model is recorded in Chroma collection metadata.** A mismatch raises `IndexMismatch`. Changing embeddings or the chunker requires `nextchap fix --rebuild-index`.
- **Resetting the index** must go through `VectorStore.reset()`. Chroma caches clients per path in-process, so deleting the folder directly breaks later opens.

## User-facing CLI surface

`nextchap` (no args) opens the book/action menu. The visible verbs are
`import`, `generate`, `status`, `fix`, `export`, `models`, `doctor`, `config`.
Book arguments are optional and resolved through `cli/resolve.py`
(exact → prefix → fuzzy → picker). `books`, `state`, `index`, `clean` and
`export epub` remain hidden legacy aliases. `[defaults]` in `nextchap.toml`
sets recurring choices (`chapters`, `max_cost`, `review`, `preview`, `critic`,
`export_epub`); CLI flags always win.

## Conventions

- Python ≥ 3.12, `from __future__ import annotations`, full type hints (mypy strict), pydantic v2 for anything persisted or exchanged with a model, dataclasses for simple internal records.
- Ruff formats at 100 columns. Lint allows up to 120 for long strings.
- User-visible failures raise a `NextChapError` subclass with a `hint`. `cli/app.py::main` renders them as panels with exit code 1. Unexpected exceptions are logged to `<data>/logs/nextchap.log`.
- UI only goes through the `Reporter` protocol (`pipeline/reporter.py`). The engine must not import Rich. Results go to stdout (`AppContext.out`) and progress and logs go to stderr (`AppContext.err`).
- Prompts live in `prompts/*.md`. Keep the stable context in `prefix=` (it's cache-marked for Anthropic) and the volatile parts in the user message.

## Adding things

- **New CLI command:** add it in `cli/app.py` (or `commands_extra.py`) and add an end-to-end case in `tests/test_cli.py`.
- **New persisted field:** give it a default so old files still load. If the shape changes, add a migration (see `memory/world_state.py::migrate_legacy`, `ArcStore.load`, `Book.meta`).
- **New model role:** add it to `ModelsConfig` plus `roles()`, update `pipeline/estimate.py`, and update the README.

## Don't

- Commit book data, `.env`, or any `.epub` (all git-ignored).
- Print or log API keys. `config show` masks them, and `config set api_key` is refused on purpose.
- Make real API calls in tests.
