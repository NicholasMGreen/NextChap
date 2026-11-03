# Changelog

## Unreleased

### Humanized prose

- The writer system prompt now carries voice rules (`prompts/humanize.md`): match the book's sentence lengths, word level and punctuation; vary rhythm hard; cut negation pivots, significance inflation, hedge stacks, aphorism closers and other rhetorical scaffolding; keep the book's own irregularities (fragments, And/But openers, comma splices at speed, dialogue interruptions) but never typos, homophones or drifted tense.
- New `pipeline/humanize.py` adds deterministic checks for stock phrasing, flat sentence rhythm, and secondary convergence (one non-function word opening 30% or more of a chapter's non-function sentence openers — the crutch that replaces the one you banned). Findings join the automated checks the critic sees, so a machine-sounding scene can be flagged and rewritten under the same rules.
- Paragraph texture and over-correction are covered by the rules rather than by a count: vary paragraph length, do not close every paragraph on a beat, and do not strain to look human. Pronouns, articles and And/But openers are ignored by the opener check, so a book's normal voice never trips it.
- `generation.humanize` (default `true`) turns the rules off for books whose voice is deliberately synthetic.
- `style_default.md` bans more overused phrases, which `banned_phrases()` enforces automatically, and asks for paragraph-length variance. Existing `style.md` files are untouched.
- `PROMPT_VERSION` 2.0.0 to 2.1.0. `docs/humanize.md` records the rationale and what was deliberately left out.

## 2.0.0 — 2026-10-29

A rewrite as an installable package with a single `nextchap` command. See `docs/MODERNIZATION_PLAN.md` for the rationale.

### Models

- Budget-only, per-role routing via OpenRouter, each model falling back to the other:
  - writer and critic: Claude Haiku 5.5 (thinking off);
  - planner: DeepSeek V4.1 Flash (low reasoning);
  - analyst: DeepSeek V4.1 Flash (no reasoning).
- Live model registry: only parameters a model supports are sent, and live prices drive cost estimates.
- Structured outputs (strict JSON Schema) for plans, world-state deltas and critic reports, with a validation-and-repair pass. If the model lacks native support, the schema goes in the prompt instead.
- Embeddings: Qwen3-Embedding-0.6B by default, replacing bge-large-en-v1.5. A remote backend is available for installs without torch. An optional Qwen3 reranker is included.

### Hardening

- Book IDs are validated. Previously `--book-id ../x` could reach `rmtree`.
- Journaled, idempotent chapter commits. A crash at any step resumes without re-planning, re-writing, or applying a world-state delta twice.
- Atomic UTF-8 writes everywhere, and a per-book file lock.
- LLM client:
  - fails fast on 401, 402 and 403 with fix hints;
  - retries only 429, 5xx, timeouts and empty answers, honoring `Retry-After`;
  - continues prose cut off by the length limit;
  - retries content-filter refusals on the fallback model;
  - enforces `--max-cost` before every request.
- World state:
  - stable character and thread IDs;
  - deltas applied to a copy and validated;
  - snapshot after every chapter;
  - `state diff` and `state rollback [--truncate]`.
- Chapter detection:
  - headings are found from document structure and number continuity, so prose like `3. He ran…` no longer splits chapters;
  - Roman numerals, number words (`Chapter Twenty-One`), `第N章`, and prologue/epilogue headings are supported;
  - numbering restarts are renumbered with a warning;
  - an anomaly report is shown before any spend.
- Fixed duplicate-header whitespace normalization: the old regex matched a literal backslash.
- EPUB export keeps the cover and front matter, adds NAV and NCX, keeps `#` and `*` in prose, and renders `*italic*`, `**bold**` and scene breaks.
- Edited chapters in `master_book.md` are detected by content hash, then re-summarized and re-indexed. Appended chapters are folded into the world state.

### Generation

- One structured plan, written scene by scene with streaming. This replaces the duplicated "Part 2" plan and write passes.
- The stable context (style guide, bible, world state, recent chapters) is prompt-cached across scene calls.
- Hybrid dense + BM25 retrieval over scene-sized chunks, with per-scene queries.
- Continuity critic plus local checks (length, repetition, banned phrases from `style.md`). Major issues trigger one targeted scene rewrite.
- Generated chapters get real titles from the plan.

### CLI

- Typer + Rich:
  - live progress for each phase;
  - streaming prose preview;
  - cost, token and cache-hit footer;
  - `--dry-run` cost tables;
  - `--review` to accept, edit in `$EDITOR`, regenerate, or quit;
  - `--json` output and `--plain` mode;
  - shell completion for book IDs.
- New commands: `models`, `doctor`, `config`, `migrate`, `state show/diff/rollback`, and `index rebuild`, which implements the documented but missing `rebuild-chroma`.
- `generated` count no longer double-counts part-2 files.

### Breaking changes

- Data moved from `./novel_books` and `./chroma_books` into the user data directory. Run `nextchap migrate`, and the index is rebuilt locally.
- Requires Python ≥ 3.12. 1.x claimed 3.10, but the code didn't parse before 3.12.
- `recent_chapters.json` is no longer used, and the `_part2*.md` artifacts are gone.
- Continuing from a merged `*_with_generated.epub` is replaced by editing or appending to `master_book.md`.
- `novel_generator.py` and `book_manager.py` remain as deprecated shims for one release.
