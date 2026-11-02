# BUGS.md — issues found while locating the `Deadpool_Twilight_V2` import

Found on 2026-11-02, CLI v checked via `uv run nextchap --help`.

## 1. Unknown subcommand under a group is reported as an "Unexpected error" (CLI error handling) — FIXED

**Status:** fixed in `cli/app.py::main`.

**Repro (before fix):**

```bash
uv run nextchap books show deadpool_twilight_v2
```

There is no `show` subcommand (the correct one is `books info`), but instead of a
friendly usage message, the CLI prints a 60-line raw traceback wrapped in an
"Unexpected error" panel and writes it to the log file.

**Cause:** `cli/app.py::main` calls the Typer app with `standalone_mode=False`,
so click's `UsageError` (user error — wrong subcommand) escaped the user-facing
`NextChapError` handler and fell into the generic "unexpected exception"
handler that dumps full tracebacks. Extra wrinkle found while fixing: Typer
≥ 0.20 vendors its own copy of click (`typer._click`), so the existing
`isinstance(exc, click.ClickException)` check could never match Typer's
`UsageError` even if it had been caught. `main()` now checks both the installed
click's and Typer's vendored `ClickException`, prints the normal one-line usage
error and returns exit code 2, with nothing logged.

**Expected (now actual behavior):** a short usage line, e.g.
`Error: No such command 'show'.` with `Try 'nextchap books -h' for help.`,
exit code 2, no traceback, no log write.

## 2. Legacy 1.x data at repo root is easy to mistake for the real book store (UX / doc gap) — RESOLVED by design change

**Resolution:** per owner request, the new layout now uses the 1.x location.
Books are stored in `<project>/novel_books/<book_id>/` again; the default data
root is the project (working) directory, no longer platformdirs
(`config.py::data_root`, `storage/paths.py::books_dir`). Overrides
`--data-dir` / `NEXTCHAP_HOME` still work (books go into `<root>/novel_books`).
The misleading "Found 1.x book data in ./novel_books" banner was removed
(it would have fired on every run now that `novel_books/` is the live store).
The books previously stored in
`~/Library/Application Support/nextchap/books/` were copied into
`./novel_books/` (the old location is left in place as a backup and can be
deleted manually).

**Original observation (kept for the record):** the repo root contained the old
1.x layout (`./novel_books/`, `./chroma_books/`) while the CLI stored books in
platformdirs — two book stores, and the imported book was "lost". This cost a
confused user session and is likely to repeat for any cloned project.

## 3. Bare `nextchap` command ran stale, non-editable code (tooling)

**Observation:** `nextchap` on PATH (`~/.local/bin/nextchap`) was a `uv tool`
snapshot installed from the repo on Oct 30. It kept running the old 2.0.0 code
and old storage behavior regardless of edits in the working tree, which made
diagnosing the storage confusion harder.

**Fix:** reinstalled editable:

```bash
uv tool install --force --editable /Users/ngreen/Projects/NextChap
```

`nextchap` now always executes the current repo code. For a fresh git clone,
`uv tool install --editable .` from the clone gives the same result.

## 4. Live preview "strobes" during streaming generation (UI) — FIXED (first pass)

**Status:** fixed with a throttle, per owner preference — the panel aesthetic
is unchanged; it just repaints far less often. `RichReporter.preview()` now
ignores per-token updates and only accepts the text when the number of
newlines has grown (one panel update per completed line),
`tests/test_cli.py::test_preview_updates_only_on_new_lines` covers it.

**Repro (before fix):** `nextchap generate BOOK -n N` — the live prose preview flickers like a
strobe light every time words appear.

**Cause (code read, `cli/ui.py` + `pipeline/engine.py`):**
- The engine calls `Reporter.preview()` once per streamed SSE delta (per token).
- `RichReporter` holds a `rich.live.Live(_Dashboard)` at `refresh_per_second=8`;
  the dashboard re-renders the **entire** block every frame: progress bar +
  a 12-line `Panel` whose text is the *tail* of the prose (`splitlines()[-12:]`)
  re-wrapped with `overflow="fold"`, + footer.
- Rich's Live does not diff lines — it rewrites the whole renderable block each
  refresh. As words stream in, the tail window scrolls and re-wraps, so all
  ~16 lines change almost every frame → full-block repaint 8×/s → strobe,
  especially on macOS Terminal.app.

**Fix (implemented):** throttle updates in the reporter, keep the aesthetic.
The planned fallback (sentence-scroll into scrollback) is not needed unless the
strobe persists on some terminals; note that rich's Live rewrites identical
lines every tick (spinner, elapsed time), so if any residual flicker shows up
next to the preview, raising `refresh_per_second` down or freezing the spinner
are the next knobs.

## 5. Non-bugs worth knowing (no action needed)

- `books info` shows `Title: Fake Deadpool In Twilight` — that string is the
  literal `<dc:title>` inside `Deadpool_Twilight_V2.epub`'s `content.opf`,
  i.e. the author's own metadata, not something nextchap mangled. If the
  display title bothers you, the source epub metadata needs fixing (or a
  `book.json` title override could be added as a feature).
- `books list` shows identical sizes for all books — normal, the three books
  are similar in size.
