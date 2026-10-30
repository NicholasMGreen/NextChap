# NextChap Modernization Plan

Status: **implemented in 2.0.0** (branch `modernize`) · Written 2026-10-29 · Model data
verified against OpenRouter, Anthropic docs, LMArena, and the MTEB leaderboard on that date.

## Implementation notes (2026-10-29)

All six phases landed together. Where the implementation differs from the plan:

- **HTTP client:** a direct `httpx` client against OpenRouter's REST API is used instead
  of the `openai` SDK. openai 3.x is built on a forked `httpx2`, and most of what the
  client needs is OpenRouter-specific: `usage.cost`, `reasoning`, `provider`, `models`
  fallbacks, and mid-stream error events. Retries still use `tenacity`.
- **Hybrid search:** BM25 is a small in-process implementation (`retrieval/bm25.py`)
  fused with Chroma dense results via RRF. This is the fallback §2.3 allowed. Chroma's
  `Search`/`Rrf` API and the `bm25s` dependency are not used.
- **Prices:** the live OpenRouter price for DeepSeek V4.1 Flash is $0.15 in and
  $0.60 out per million tokens, not the $0.044 / $0.383 found by search. The estimates
  still hold: about $0.01 per chapter, and about $0.25 to import a 500-chapter book
  (`generate --dry-run`).
- **Haiku parameters:** Haiku 5.5 does not accept `temperature` or `top_p`. The client
  filters parameters by each model's `supported_parameters`, so they are never sent.
- **Layout:** the registry lives at `nextchap/models.py`, the engine at
  `pipeline/engine.py`, and the no-LLM operations at `pipeline/manage.py`.
- **Not yet verified against live models:** no API key was available during
  implementation. These need a live run to confirm:
  - the Phase 4 cache-hit target;
  - the Phase 3 name-recall eval on a real book.

  Everything else is covered by 86 offline tests, which include crash-injection resume
  tests and pass on Python 3.12 and 3.14.

Goal: bring NextChap up to current models and current Python practice. Make
the pipeline crash-safe and correct, and replace the `print()` output with a
proper terminal UI. It stays a CLI with no web app.

---

## 1. Where the project is today

Two scripts make up the whole project. `novel_generator.py` is about 1,550 lines
and `book_manager.py` is about 570. The stack was chosen in early 2025:

| Concern | Today | Problem |
|---|---|---|
| LLM | `deepseek/deepseek-chat` (DeepSeek V3), one model for every task | Two generations old. No routing by task. The cost table in the README is stale. |
| Embeddings | `BAAI/bge-large-en-v1.5` via LlamaIndex + torch | No longer competitive on MTEB English v2. 512-token window. Pulls in about 2 GB of torch. |
| Retrieval | Dense-only top-k, query is the last hook plus four names | Weak on proper nouns such as character, technique, and place names, which are the main thing a web novel needs. |
| JSON extraction | Prompt "respond only JSON", strip ``` fences, `json.loads` | No schema enforcement. Silent fallbacks. |
| Packaging | `requirements.txt` with `>=` pins, no lockfile | Builds can't be reproduced. |
| CLI/UI | `argparse` and `print()` with `\r` progress | No live progress, cost display, or streaming. |
| Tests/CI | None | Every change is a gamble on a paid pipeline. |

### 1.1 Confirmed bugs (fix in Phase 1)

Line numbers refer to the current `master` branch.

1. **Path traversal in `--book-id`.** Book IDs go straight into
   `BOOKS_ROOT / book_id` (`novel_generator.py:1546`, all `book_manager` commands),
   and `delete` calls `shutil.rmtree` on the result. Validate IDs against
   `^[a-z0-9][a-z0-9_-]{0,63}$`.
2. **Python version mismatch.** `novel_generator.py:191` has a backslash inside
   an f-string expression. That is a `SyntaxError` before Python 3.12, but the
   README says 3.10+.
3. **Broken whitespace normalization.** The same line uses `r'\\s+'`, which
   matches a literal backslash. Verified: `"The   Beginning"` stays unnormalized,
   so duplicate-header detection misses headers that differ only in spacing.
4. **False chapter splits.** In `TextParser._PATTERNS` (`:470-475`),
   `^(\d+)[.:]\s+(.+)$` matches prose such as `3. He ran to the door.` and
   `1984: the year it began` (both verified). Roman-numeral headers are not
   matched. `EpubHandler._HEADER_PATTERNS` (`:261`) is dead code.
5. **Vector dedupe most likely never matches.** `index_chapters()` (`:887`)
   compares against `chapter_N` IDs, but LlamaIndex writes chunk (node) IDs. A
   re-run after a crash duplicates vectors.
6. **Crash window corrupts state.** `generate_chapters()` appends to
   `master_book.md` (`:1439`) and saves `world_state.json` (`:1442`) before
   `_save_progress()` (`:1468`). If the process dies in between, the next resume
   treats the generated chapter as new source, re-summarizes it, applies its
   world-state delta a second time, and re-indexes it.
7. **LLM client.** `LLMClient.complete()` (`:526-558`) has several problems:
   - It retries every exception, including 401/402/400.
   - It sets no timeout.
   - `message.content` can be `None`, and `.strip()` then crashes.
   - It ignores `finish_reason == "length"`, so a truncated chapter is accepted
     as complete.
   - It never reads `usage`, so cost is invisible.
8. **World-state delta is not atomic.** `update_delta()` (`:808-852`) mutates
   `self.state` in place. `s.characters[name].update(updates)` crashes when the
   value isn't a dict. Thread removal needs an exact string match, so threads
   are almost never resolved. In `update_safely()`, `old_state` is captured and
   never used.
9. **Lossy EPUB export.** `_clean_for_epub()` (`:401`) deletes every `*` and `#`,
   which turns "Rank #1" into "Rank 1". `write()` (`:370-376`) wipes the spine
   and TOC and never adds `EpubNav`/`EpubNcx`, and it drops the cover and front
   matter pages.
10. **Writer prompt mismatch.** The writer receives `chapter_num="N Part 2"`, a
    string, and every generated chapter is titled "Chapter N" with no real title.
11. **Missing command.** `rebuild-chroma` is documented in the README
    (`README.md:218-222`) but not implemented.
12. **Generated count is inflated.** `book_manager._count_generated()` counts
    `_part2.md` files, so the number is roughly double.
13. **Writes can corrupt files.** There are no atomic writes and no explicit
    `encoding="utf-8"`, so a crash mid-write corrupts JSON or markdown. Nothing
    stops two concurrent runs on the same book.
14. **Duplicated logic drifts.** Cleanup and header regexes are copied between
    the two files, and the two `_CHAPTER_HEADER_RE` variants already differ.
15. **API key exposure.** `--api-key` leaks the key into shell history and the
    process list (`ps`).

---

## 2. Target model stack (October 2026)

### 2.1 Principles

- **Route models by role.** Analysis (summaries, world-state extraction) is
  high volume and needs fidelity, not flair, so it goes to a cheap model.
  Planning needs reasoning. Prose needs voice.
- **Turn reasoning down or off for prose.** Reviewers report that thinking
  modes hurt line-level prose. Reasoning models also spend `max_tokens` on
  hidden thinking: DeepSeek V4.1 Flash and Gemini 3.8 Flash returned *empty
  strings* in October fiction tests at default settings. Every call sets
  `reasoning` explicitly, and the client treats an empty answer as a retryable
  failure.
- **Keep the continuity checker separate from the writer.** The check runs as
  an independent call that sees only the finished chapter and the story
  records, never the writer's prompt or plan reasoning.
- **Keep model IDs in config, never in code.** At startup, validate the
  configured slugs against `GET https://openrouter.ai/api/v1/models`. Use each
  model's `supported_parameters` so the client only sends what that model
  accepts (temperature, `top_p`, `reasoning`, `response_format`).

### 2.2 Model configuration (budget only)

NextChap ships one model setup, tuned for low cost. There are no presets or
tiers. Each role's model is set in `nextchap.toml`, and the defaults are:

| Role | Model | Settings |
|---|---|---|
| Writer (prose) | `anthropic/claude-haiku-5.5` | thinking off |
| Planner | `deepseek/deepseek-v4.1-flash-1017` | reasoning effort `low` |
| Analyst (summaries, world state, structured JSON) | `deepseek/deepseek-v4.1-flash-1017` | reasoning `none` |
| Continuity critic | `anthropic/claude-haiku-5.5` | thinking off |

Why these models:

| Model | OpenRouter slug | $/M in → out | Context | Notes |
|---|---|---|---|---|
| Claude Haiku 5.5 | `anthropic/claude-haiku-5.5` | 0.10 → 0.50 (≤100K prompt; cache read 0.01) | 1M / 128K out | Held every planted story fact in October fiction tests for a fraction of a cent. Thinking can be turned off, which suits prose. Prompts over 100K tokens cost 5× more, so the context budget stays under that. |
| DeepSeek V4.1 Flash | `deepseek/deepseek-v4.1-flash-1017` | 0.044 → 0.383 | 1M | The direct successor to the current DeepSeek V3 choice and the most-used model on OpenRouter. Supports JSON structured outputs. Reasoning **must** be disabled or capped, or it can return empty answers. |

The writer and the critic are both Haiku. The critic runs as an independent
call with its own prompt (see §2.1), so no third model is needed.

Other models are out of scope for this plan. Model slugs live in config, so
changing a role later only needs a config edit, and the config accepts any
OpenRouter slug.

**Rough cost per generated chapter.** Assumptions: two plan calls, two or
three write segments, one summary, one world-state delta, and one critic call.
That is about 45K input and 13K output tokens before prompt caching. The
estimate is **about $0.01 per chapter**.

Initial ingestion uses the analyst model only, which comes to roughly $0.20
for a 500-chapter book. The CLI shows a real pre-run estimate and enforces a
`--max-cost` cap (see §4).

### 2.3 Embeddings, retrieval, reranking

| Option | MTEB eng v2 | Size | Notes |
|---|---|---|---|
| **`Qwen/Qwen3-Embedding-0.6B`** (new default) | 70.47 | ~1.8 GB RAM | Apache-2.0, 32K-token input, 1024 dimensions. Runs on CPU or Apple MPS. |
| `Qwen/Qwen3-Embedding-4B` (`--embed large`) | 74.61 | ~5.3 GB | For machines with a GPU. |
| `google/embeddinggemma-300m` (`--embed small`) | 69.67 | ~1.2 GB | Small option. License has usage conditions. |
| Remote (`--embed remote`) | n/a | 0 | Embeddings through an OpenAI-compatible API, so users can skip installing torch. |

Retrieval changes:

- **Drop LlamaIndex.** It is only a thin wrapper here, and it hides the node
  IDs that caused bug 5. Use `chromadb>=1.5` directly.
- **Chunk by scene.** Split on scene breaks or paragraph windows of about 400
  words with overlap, instead of embedding whole chapters. Use deterministic
  IDs (`ch{N:05d}-s{K:03d}`) and `upsert`, so re-indexing is idempotent.
- **Use hybrid search.** Combine dense and BM25 results with reciprocal rank
  fusion (RRF), so exact names like "Azure Dragon Fist" are found. Chroma 1.3+
  ships BM25 sparse indexes and the `Search`/`Rrf` API. *Verify that it works
  on local `PersistentClient` before relying on it.* The fallback is in-process
  `bm25s` merged with RRF in our code.
- **Rerank (optional).** Use `Qwen/Qwen3-Reranker-0.6B` on the top 30 results,
  keep the top 6, and enable it with `--rerank`.
- **Build better queries.** Use the planner's scene list (characters,
  location, objects in play) for each scene, not just the last hook.
- **Record the embedding setup.** Store the model name, dimensions, and chunker
  version in collection metadata. On mismatch, refuse to query and tell the
  user to run `nextchap index rebuild`.

---

## 3. Hardening the logic

### 3.1 LLM client (`nextchap/llm/client.py`)

- Use `openai>=2` with `base_url` set to OpenRouter. Pass the official
  `X-OpenRouter-Title` / `HTTP-Referer` headers, an explicit `timeout`, and
  `max_retries=0`, because retries are handled by our own policy.
- Retry with `tenacity`. Retry only on `RateLimitError`, `APITimeoutError`,
  `APIConnectionError`, 5xx, and the client's own `EmptyCompletion`. Use
  exponential backoff with jitter and honor `Retry-After`. Fail immediately on
  401, 402, and 400 with a clear message.
- Check `finish_reason`:
  - `length` on prose triggers a continuation request ("continue exactly where
    you stopped"), up to two extra segments.
  - `length` on JSON is a hard error.
  - `content_filter` is surfaced to the user and retried once on the other
    configured model.
- Request `usage: {include: true}` so OpenRouter returns the cost of each call.
  Keep a `CostMeter` per run and per book, persisted to `usage.jsonl`, and
  enforce `--max-cost`.
- Fall back through OpenRouter's `models: [...]` list. Haiku 5.5 and
  DeepSeek V4.1 Flash back each other up, so an outage at one provider doesn't
  stop a run.
- Stream prose (`stream=True`) so the UI can show text as it arrives and a
  stalled connection is noticed quickly.
- Cache stable prefixes. Order every prompt as stable prefix first (system,
  style guide, story bible, world-state snapshot), then the volatile suffix
  (plan, recent prose). Put a `cache_control` breakpoint after the prefix.
  Haiku 5.5 cache reads cost 10% of the normal input price, and the bible is
  resent on every call.

### 3.2 Structured outputs

- Define every JSON artifact as a Pydantic v2 model: `ChapterSummary`,
  `WorldState`, `WorldStateDelta`, `ChapterPlan`, `CriticReport`.
- Send `response_format={"type": "json_schema", "json_schema": {"strict":
  true, ...}}`, generated from the model, with
  `provider: {require_parameters: true}`.
- Validate with `Model.model_validate_json`. On failure, do one repair call
  that includes the validation error, then fail loudly. Don't fall back
  silently to an empty state.
- The `ChapterPlan` includes a `title`, which fixes the "Chapter N" titles.

### 3.3 World state

- Give characters and threads stable IDs (`char_lin_feng`, `thread_007`). Delta
  operations refer to IDs, not free text, so removing a thread actually works.
- Apply deltas to a **deep copy**, validate the result (no mass deletion, hook
  present, chapter number increases), and only then swap it in.
- Keep a version history: `world_state/ch0123.json` snapshots plus a `diff.log`.
  A new `nextchap state rollback <book> --to 120` command restores a snapshot.
- Add a `schema_version` field and a loader that migrates v0 files (the
  current dataclass dump) forward.

### 3.4 Crash-safe chapter commits

Each chapter is committed in four steps:

1. Write everything for the chapter into `staging/chNNNN/`: plan, prose,
   summary, delta, and critic report.
2. Write a journal entry `journal.json`:
   `{chapter, step, model, prompt_hash, cost}`.
3. In order, append to `master_book.md`, write the new world state, upsert the
   vectors, write the summaries, then mark the journal entry `committed`. Every
   file write goes through `atomic_write()`, which writes a temp file, calls
   `fsync`, then `os.replace`. All I/O uses `encoding="utf-8"`.
4. On startup, an uncommitted journal entry is either finished (if staging is
   complete) or discarded. No step runs twice, and vector upserts are
   idempotent.

Lock each book with a `filelock` on `novel_books/<id>/.lock` so two runs can't
interleave. Make Ctrl-C safe: the first SIGINT finishes the current step,
commits, and exits, and the second aborts.

### 3.5 Parsing and EPUB

- Put the parsing and cleanup code in one module, `nextchap/ingest/`, used by
  both the generator and the book manager.
- A line only counts as a chapter header when all of these hold:
  - it is short (≤ 80 characters) and stands alone (blank lines or a block
    boundary around it);
  - for EPUB input, it is preferably an `<h1>`–`<h3>` element or the first
    line of a spine item;
  - its number continues the sequence (n+1, allowing small gaps), so prose
    like `1984:` is rejected.
- Support Roman numerals, `Chapter One`, and `第12章`.
- Print a summary of what was detected and show anomalies (gaps, duplicates,
  very short chapters) before any money is spent. `--yes` skips the prompt.
- Move the promo and noise patterns into a configurable `cleanup.toml` list.
- EPUB export:
  - keep the cover and front matter;
  - add `EpubNav` and `EpubNcx`;
  - strip markdown emphasis and heading markers only, not every `*` or `#`;
  - convert `*italic*` to `<em>`;
  - escape with `html.escape`;
  - optionally validate with `epubcheck` when it is on PATH
    (`nextchap export epub --check`).

### 3.6 Generation quality gates

- **Single plan.** One `ChapterPlan` (structured, with scenes and beats)
  replaces the duplicated `plan` / `plan_part_two` prompts. The writer then
  writes scene by scene and streams each segment. Length is controlled by
  counting words and continuing, not by asking twice.
- **Context budget.** A `ContextBuilder` allocates a token budget per section
  (bible, world state, recent prose, RAG hits) using `tiktoken`-style
  estimates, instead of fixed character slices like `[:8000]` or `[:1800]`.
  Writer prompts are capped below 100K tokens, where Haiku 5.5's price
  rises 5×.
- **Critic pass.** A different model checks continuity against the world state
  and recent summaries, checks that the plan was followed, and flags repeated
  phrases and word count. The output is a `CriticReport` with a severity. On
  `major`, rewrite the flagged scenes once, then keep the chapter and record
  the warning.
- **Style guide.** Keep a user-editable `style.md` (POV, tense, banned phrases,
  dialogue conventions) that sits in the cached prefix of every writer prompt.
- **Prompt files.** Move prompts to versioned `nextchap/prompts/*.md` templates
  instead of inline strings. Record the prompt version in each chapter's
  metadata.

### 3.7 Security and configuration

- Read the API key from env, then `.env`, then the OS keyring (`keyring`).
  `--api-key` stays only as a hidden, deprecated flag. Redact keys in logs and
  tracebacks.
- Settings come from `pydantic-settings`, in this order: CLI flags, env
  (`NEXTCHAP_*`), the project file `nextchap.toml`, then the user config in the
  platformdirs config directory.
- Put the data root at `platformdirs.user_data_dir("nextchap")`, overridable
  with `--data-dir` / `NEXTCHAP_HOME`, instead of relative to the working
  directory. A one-time migration detects an old `./novel_books` and offers to
  move it.

---

## 4. Terminal UI (Typer + Rich, still a CLI)

All commands live under a single `nextchap` entry point. Old script names keep
working through thin shims for one release.

```
nextchap import  <file> [--book-id ID] [--yes]
nextchap generate <book-id> -n 3 [--max-cost 0.50] [--review] [--dry-run]
nextchap books list | info <id> | delete <id> [--yes]
nextchap clean <id>            # was: book_manager cleanup
nextchap state rebuild <id> | rollback <id> --to N | show <id>
nextchap index rebuild <id>    # implements the documented rebuild-chroma
nextchap export epub <id> [--output] [--template] [--check]
nextchap models [--refresh]    # model per role, live prices, availability
nextchap doctor                # key, Python, deps, disk, model reachability, embed cache
nextchap config show | set key value
```

UI behavior:

- **Progress.** A `rich.progress` display shows nested tasks: the overall
  phase, the chapter, and the current step. Each step shows a spinner, elapsed
  time, and ETA. During ingestion, the summaries bar shows
  `ch 132/540 · 3.1/s · $0.04`.
- **Live prose preview.** While the writer streams, a `Live` panel shows the
  tail of the chapter, with a word counter that turns green once the target
  length is reached. Disable it with `--no-preview`.
- **Cost footer.** A status line shows tokens in/out, cache hit rate, spend for
  this run against `--max-cost`, and the model per role.
- **Dry run.** `--dry-run` prints a table of the planned calls per role with
  token and cost estimates, and calls no models.
- **Review mode.** `--review` pauses after the plan, renders it, and offers
  `[a]ccept · [e]dit in $EDITOR · [r]egenerate · [q]uit`. After the chapter is
  written, it offers accept, regenerate the flagged scenes, or edit.
- **Tables and trees.** `books list` is a Rich table. `books info` shows
  panels plus a `Tree` of characters, threads, and factions. `state show`
  renders a world-state diff with colors.
- **Errors.** Errors appear as a Rich panel with the cause and a concrete fix,
  such as "402 Insufficient credits → top up at openrouter.ai/credits". Full
  tracebacks are written to `nextchap.log` and printed to the terminal only
  with `-v`.
- **Script-friendly output.** Logs and progress go to stderr and results go to
  stdout. `--json` gives machine output for `list`, `info`, `models`, and
  `doctor`. `NO_COLOR` and non-TTY output are respected and fall back to plain
  lines.
- **Shell completion.** Book IDs complete in the shell via Typer.

Out of scope: a full-screen Textual TUI. It can be revisited after Phase 4
without changing the core.

---

## 5. Target layout and tooling

```
pyproject.toml            # uv, requires-python >=3.12, [project.scripts] nextchap
uv.lock
src/nextchap/
  cli/        app.py  generate.py  books.py  state.py  export.py  ui.py
  config.py   models.py
  llm/        client.py  cost.py  schemas.py
  ingest/     epub.py  text.py  cleanup.py  master_book.py
  memory/     world_state.py  summaries.py  bible.py
  retrieval/  embed.py  store.py  rerank.py  context.py
  pipeline/   planner.py  writer.py  critic.py  orchestrator.py  journal.py
  storage/    paths.py  atomic.py  lock.py  migrate.py
  prompts/    *.md
tests/        unit/  integration/  fixtures/ (tiny public-domain epub + txt)
```

Dependencies:

- core: `typer`, `rich`, `openai>=2`, `pydantic>=2`, `pydantic-settings`,
  `tenacity`, `chromadb>=1.5`, `ebooklib`, `beautifulsoup4`, `lxml`,
  `platformdirs`, `filelock`, `bm25s`
- extras: `[local-embed]` adds `sentence-transformers` and `torch`;
  `[rerank]` adds the reranker; `[dev]` adds `pytest`, `pytest-cov`, `respx`,
  `ruff`, `mypy` (or `ty`), and `pre-commit`

Tooling:

- Use `ruff` for lint and format and `mypy --strict` on `src/`.
- Run pytest in CI (GitHub Actions, Python 3.12 to 3.14, Linux and macOS).
- Tests never hit the network. LLM calls go through a `FakeLLM` that replays
  recorded fixtures.

---

## 6. Phased roadmap

Each phase is mergeable on its own, and existing books keep working after each
one.

| Phase | Scope | Done when |
|---|---|---|
| **0. Safety net** | Add `pyproject.toml`, uv, ruff, and pytest. Add characterization tests for the parser, cleanup, markdown round-trip, delta merge, and EPUB round-trip on fixtures. | CI is green and the tests pin current behavior, bugs included, marked `xfail`. |
| **1. Correctness** | Fix bugs 1–15 from §1.1 inside the current files. Add atomic writes and utf-8, book-ID validation, the LLM client fixes, the journaled commit, the file lock, `rebuild-chroma`, and the generated-count fix. | The `xfail`s pass. Killing the process at any step and resuming produces the same state as an uninterrupted run (tested with fault injection). |
| **2. Package and models** | Move to the `src/nextchap` layout. Add config with the budget models from §2.2, the model registry validated against `/models`, per-role routing, Haiku/DeepSeek fallback, explicit reasoning settings, structured outputs, cost metering, and `--max-cost`. | `nextchap generate` works end to end at about $0.01 per chapter, and no call returns an empty answer because of reasoning. `doctor` and `models` work. Old scripts still run through shims. |
| **3. Retrieval** | Drop LlamaIndex. Add the Qwen3 embeddings, scene chunking, hybrid BM25 + RRF, the optional reranker, embedding metadata guards, and `index rebuild`. | Name-recall eval on fixtures (correct chapter in top 6 for 20 named-entity queries) is ≥ 90%, up from the measured baseline. |
| **4. Generation quality and UI** | Add the single structured plan, scene-wise streaming writer, critic pass, style guide, prompt files, prompt caching, the Rich UI, review mode, dry run, and `--json`. | Each chapter shows live progress, preview, and cost. Cache hit rate on Haiku writer calls for chapters 2+ is ≥ 60%. |
| **5. Docs and migration** | Rewrite the README for Linux and macOS. Add `nextchap migrate` for old `./novel_books` data, update AGENTS.md, and write a CHANGELOG. | A fresh clone reaches a first chapter with `uv tool install .` plus 3 commands. |

Suggested order inside Phase 1, from least to most effort: bugs 1, 13, 7, 6,
then the rest.

---

## 7. Open questions for the owner

1. **Content policy.** Do the target books need mature or dark content that
   Haiku may soften? If so, the writer role can be pointed at DeepSeek V4.1
   Flash (reasoning off) in config without other changes.
2. **Embedding locality.** Keep local embeddings as the default (needs about a
   2 GB download), or default to remote embeddings so the install stays light?
3. **Keep the two-part chapter structure?** The plan replaces it with
   scene-wise writing and the same total length. Confirm that nobody depends
   on the `_part2*.md` artifacts.
4. **Migration.** Is it acceptable to re-index every existing book once? This
   is required by the embedding change and is local-only, so it costs no API
   money.
