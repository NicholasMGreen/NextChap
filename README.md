# NextChap

Continue a web novel from an existing `.epub` or `.txt`. NextChap builds a long-term memory of the story, then plans, writes, checks and commits new chapters. It runs entirely in the terminal.

- **Writer:** Claude Haiku 5.5 with thinking off. **Planner and analyst:** DeepSeek V4.1 Flash. All calls go through [OpenRouter](https://openrouter.ai), and each model falls back to the other.
- **Story memory:**
  - chapter and arc summaries;
  - a structured world state with stable character and thread IDs, plus a snapshot per chapter;
  - an editable story bible and style guide.
- **Retrieval:** local Qwen3 embeddings with hybrid dense + BM25 search over scene-sized chunks, so exact names like "Azure Dragon Fist" are found.
- **Safety and quality:** a crash-safe, resumable commit journal, a spend cap, and a continuity critic that can revise flagged scenes.
- **Output:** EPUB that keeps the original cover and metadata and adds a proper table of contents.

Typical cost is **about $0.01 per generated chapter** and **about $0.25 to import a 500-chapter book**. Prices come live from OpenRouter, and `--dry-run` shows the estimate before you spend anything.

---

## Install

You need Linux or macOS, Python 3.12+, [uv](https://docs.astral.sh/uv/), and an OpenRouter API key.

```bash
git clone <this repo> && cd NextChap
uv tool install ".[local-embed]"       # installs the `nextchap` command
# or, for development: uv sync --extra local-embed && uv run nextchap --help

export OPENROUTER_API_KEY=sk-or-...    # or put it in .env, or: nextchap config set-key
nextchap doctor                        # checks key, credits, models, embeddings, disk
```

`local-embed` pulls in PyTorch, and the embedding model (about 1.2 GB) downloads on first use. For a light install without torch, skip the extra and use remote embeddings:

```bash
uv tool install .
nextchap config set retrieval.embed_backend remote
```

## Quick start

```bash
nextchap import my_novel.epub --book-id my_story   # shows detected chapters + cost, then builds memory
nextchap generate my_story -n 3                     # plans, writes, checks and commits 3 chapters
nextchap books info my_story                        # overview, world state, total spend
```

While it runs you see a progress bar for each phase, a live preview of the prose as it streams, and a footer with spend, tokens, and cache hit rate. Press Ctrl-C once to stop at the next safe point, or twice to abort. Re-running the same command resumes from the last completed step, whether you stopped it yourself or it crashed.

Useful flags:

| Flag | Effect |
|---|---|
| `--dry-run` | Show the planned calls, tokens and cost; spend nothing |
| `--max-cost 0.50` | Never spend past this; stops between chapters when the next one would exceed it |
| `--review` | Approve, edit in `$EDITOR`, or regenerate each plan and each chapter |
| `--no-critic` | Skip the continuity check |
| `--no-preview` | Hide the streaming preview |
| `--plain` | Plain log lines (automatic when output isn't a terminal) |
| `--json` | Machine-readable output for `books list/info`, `state show`, `models`, `doctor`, `config show` |

## Commands

```
nextchap import FILE [--book-id ID] [--dry-run] [--yes] [--force] [--max-cost N]
nextchap generate BOOK [-n N] [--max-cost N] [--review] [--dry-run] [--no-critic] [--no-epub]
nextchap books list | info BOOK | delete BOOK [--yes]
nextchap clean BOOK [--no-backup]                 # strip promo lines / duplicate headers
nextchap state show BOOK [--chapter N] | diff BOOK [--from A --to B]
nextchap state rollback BOOK --to N [--truncate]  # undo world-state (and chapters) after N
nextchap state rebuild BOOK [--no-renumber] [--dry-run]
nextchap index rebuild BOOK                       # re-embed (local, no API cost)
nextchap export epub BOOK [-o PATH] [--template EPUB] [--check]
nextchap models [--refresh]                       # model per role, live price, availability
nextchap doctor
nextchap config show | path | set KEY VALUE [--project] | set-key
```

Shell completion (including book IDs): `nextchap --install-completion`.

## Configuration

Settings are layered, highest priority first: CLI flags, `NEXTCHAP_*` environment variables (nested with `__`, e.g. `NEXTCHAP_GENERATION__MIN_WORDS=1200`), `.env`, `./nextchap.toml`, then the user config (`nextchap config path`). Example `nextchap.toml`:

```toml
[generation]
target_words = [1800, 2400]
min_words = 1200
max_cost = 1.00          # default --max-cost
scene_separator = "* * *" # "" joins scenes without a break

[models.writer]
model = "anthropic/claude-haiku-5.5"
reasoning = "none"       # keep thinking off for prose
max_tokens = 6000
fallbacks = ["deepseek/deepseek-v4.1-flash-1017"]

[retrieval]
embed_backend = "local"  # local | remote
embed_model = "Qwen/Qwen3-Embedding-0.6B"
rerank = false           # true needs: uv tool install ".[local-embed,rerank]"

[cleanup]
extra_noise_patterns = ["^translator'?s note"]
```

Any OpenRouter model ID works for any role. The client reads each model's supported parameters from OpenRouter and only sends what that model accepts. Changing `retrieval.embed_model` means you need to run `nextchap index rebuild`; NextChap detects the mismatch and offers to do it.

## How it works

**Import** (once per book). NextChap first detects chapters, using heading structure and number continuity so that prose lines like `3. He ran…` aren't mistaken for headings. It then removes promo noise and shows you the result before spending anything. After you confirm, it:

1. summarizes every chapter;
2. builds arc summaries;
3. extracts a structured world state;
4. writes a story bible;
5. indexes scene chunks.

**Each new chapter.** Every step is journaled under `staging/`:

1. **Plan:** a structured scene plan with beats, thread IDs, and an ending hook.
2. **Write:** scene by scene, streaming. The stable context (style, bible, world state, recent chapters) is prompt-cached across scenes. Each scene also gets hybrid-retrieved past scenes.
3. **Check:** local checks for length, repetition and banned phrases, plus an independent critic pass. Scenes with major issues are rewritten once.
4. **Analyze:** summary plus a validated world-state delta, applied to a copy and rejected if it would lose data.
5. **Commit:** written in a fixed order: `master_book.md`, then world state, then index, then summaries, then `book.json`. Every write is atomic, and a file lock prevents two runs on one book.

`master_book.md` is the source of truth. You can edit any chapter or append chapters by hand. On the next run, changed chapters are re-summarized and re-indexed, and appended chapters are folded into the world state.

## Files

Books live in `novel_books/` inside the project directory (same layout as 1.x). A different root can be set with `--data-dir` or `NEXTCHAP_HOME`:

```
novel_books/<book-id>/
  master_book.md            canonical text of every chapter (edit freely)
  style.md                  style guide sent with every writing request (edit freely)
  master_bible.md           story bible (edit freely; refreshed every 5 chapters)
  world_state.json          characters, threads (t001…), rules, locations, factions, hook
  world_state_history/      snapshot after every chapter (used by state diff/rollback)
  chapter_summaries.json, arc_summaries.json, book.json, usage.jsonl (per-call cost log)
  generated_chapters/       chapter_NNNN.md, _plan.md, _critic.json
  <source>_with_generated.epub
  index/                    vector index
logs/nextchap.log           full debug log (tracebacks land here)
```

## Upgrading from 1.x

1.x stored books in `./novel_books` — 2.0 uses the same folder, so old imports are picked up in place, but the 1.x scripts (`novel_generator.py`, `book_manager.py`) were removed. Re-run `nextchap import …` for any book whose story memory you want rebuilt (world state, summaries and the index are regenerated locally, no API cost for the index; summaries cost a small analyst/planner pass).

## Troubleshooting

- **`doctor` says the API key failed:** check `OPENROUTER_API_KEY`. HTTP 402 means you're out of credits (top up at openrouter.ai).
- **Empty answers or "spent on hidden reasoning":** lower the role's `reasoning` setting (`none` or `low`).
- **The model refuses dark content:** point the writer role at another model, e.g. `nextchap config set models.writer.model deepseek/deepseek-v4.1-flash-1017`. A content-filter refusal is automatically retried once on the fallback model.
- **"Book is locked":** another `nextchap` process is working on that book.
- **Wrong chapter splits:** look at the anomalies list in `import --dry-run`. Fix the source or `master_book.md`, then run `nextchap state rebuild`.

## Development

```bash
uv sync                      # dev tools included
uv run pytest                # no network; uses a fake model client and hash embeddings
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

See `AGENTS.md` for architecture notes and `docs/MODERNIZATION_PLAN.md` for the design rationale.
