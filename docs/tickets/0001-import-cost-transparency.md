# Ticket: Import cost estimate looks incomplete / footer misleads about which model does the work

**Status:** done (branch `feat/estimate-ui-polish`)
**Priority:** medium
**Type:** bug (misleading UI) + small UX improvement
**Risk:** low — display-only changes, no pipeline or cost-logic changes

---

## Symptom (user report)

During `nextchap import`, the pre-flight estimate table shows only `analyst` /
`deepseek/deepseek-v4.1-flash-1017` rows:

```
Estimated import cost
 Role    ┃ Model                             ┃ Calls ┃ Input tok ┃ Cached ┃ Output tok ┃    Cost
━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━
 analyst │ deepseek/deepseek-v4.1-flash-1017 │    13 │    34,402 │      0 │      4,550 │ $0.0079
 analyst │ deepseek/deepseek-v4.1-flash-1017 │     1 │     7,250 │      0 │        900 │ $0.0016
 analyst │ deepseek/deepseek-v4.1-flash-1017 │     1 │     5,150 │      0 │      2,500 │ $0.0023
 analyst │ deepseek/deepseek-v4.1-flash-1017 │     1 │     8,000 │      0 │      2,500 │ $0.0027
─────────┼───────────────────────────────────┼───────┼───────────┼────────┼────────────┼─────────
 total   │                                   │       │           │        │            │ $0.0145
```

But the live progress footer during import shows:

```
spend $0.0132  tokens 46,555→12,824  cache 19%  writer anthropic/claude-haiku-5.5
```

The user concluded that "haiku does the summaries and many other roles while this
summary only has deepseek", i.e. the estimate looked incomplete.

## Investigation findings (do NOT skip — this corrects the premise)

**Finding 1 — Import genuinely uses ONLY the `analyst` role (deepseek). Zero haiku/writer calls.**
Every LLM call site in the import path (`Engine.sync()`) passes the `"analyst"` role:

| Import step | Code | Role |
|---|---|---|
| Summarizing chapters (1 call/chapter) | `src/nextchap/memory/summaries.py:89` `summarize_chapter` | `analyst` |
| Summarizing arcs (1 call/arc) | `src/nextchap/memory/summaries.py:106` `summarize_arc` | `analyst` |
| Writing the master bible (1 call, only if empty) | `src/nextchap/memory/summaries.py:139` `build_bible` (called from `engine.py:130`) | `analyst` |
| Extracting the world state (1 call, first import) | `src/nextchap/pipeline/engine.py:186` `world:init` | `analyst` |
| Indexing chapters | `engine.py:229` `_sync_index` | no LLM (local embeddings) |

The `writer` / `planner` / `critic` models are only used during `nextchap generate`,
never during import.

**Finding 2 — The estimate table is correct AND complete for a fresh import.**
`src/nextchap/pipeline/estimate.py::estimate_import` produces 4 rows that map 1:1 to the
call sites above:

1. Row "13 calls" = chapter summaries (`n` calls)
2. Row "1 call, 7,250 in / 900 out" = arc summaries (`ceil(n / arc_size)` calls)
3. Row "1 call, 5,150 in / 2,500 out" = master bible
4. Row "1 call, 8,000 in / 2,500 out" = world-state init

The actual run in the symptom spent $0.0132 by the arc-summarization phase (bible not
yet called), converging on the estimated $0.0145. So the estimate matches reality.

**Finding 3 — ROOT CAUSE of the confusion: the footer hard-codes the writer model.**
`src/nextchap/cli/ui.py::RichReporter.footer()` (~line 135) unconditionally appends the
*configured* writer model to the status footer during **every** operation:

```python
t.append("  writer ", style="dim").append(self.settings.models.writer.model, style="cyan")
```

During import this prints `writer anthropic/claude-haiku-5.5` even though the writer
model performs zero calls. It is a static config readout, not "who is working now" —
but that is exactly how it reads.

**Finding 4 — The estimate table is opaque about what each row covers.**
The `Role` column shows the role name, but nothing tells the reader *which pipeline
step* each row covers (chapter summaries vs arcs vs bible vs world state). Four
identical `analyst` rows with different token counts look like noise/duplication.

**Finding 5 — Minor real gaps in the estimate (acceptable, should be documented):**
- Model **fallbacks** are not modeled (each role's model falls back to the other's per
  `ModelsConfig`); a fallback run would cost differently than estimated.
- Retries on `RetryableLLMError` (429/5xx) can add cost.
- `world:delta` calls (`engine.py:247`) only happen when chapters are added after the
  initial import — they are estimated in `estimate_generation`, not `estimate_import`,
  which is correct since `import` deletes an existing book and always starts fresh
  (`cli/app.py:382-383`).

## Proposed fix

### 1. Footer: show per-role/per-model activity instead of the static writer readout
- Add `CostMeter.by_model() -> dict[str, ModelTotals]` (or similar) in
  `src/nextchap/llm/cost.py`, aggregating cost/calls/tokens per `CallRecord.model`
  (records already carry `role` and `model` — see `CallRecord` at `llm/cost.py:20`).
- Change `RichReporter.footer()` to render what actually happened in this run, e.g.:
  - `models: analyst=deepseek/deepseek-v4.1-flash-1017 (15 calls)` — roles with ≥1 call,
    each showing the model used and call count;
  - fall back to the current static `writer <model>` display only when the meter has
    no records yet (e.g. outside model-backed flows);
  - keep `spend` / `tokens` / `cache` segments unchanged.
- Do **not** show roles with zero calls (that's what misleads today).

### 2. Estimate table: label each row with the pipeline step it covers
- Add a `step: str` field to `EstimateRow` (`pipeline/estimate.py`) and set it for every
  row in `estimate_import` and `estimate_generation`:
  - `"chapter summaries"`, `"arc summaries"`, `"master bible"`, `"world state"`, etc.
- Update `_estimate_table` in `cli/app.py` to include the step column.
- This makes the 4 `analyst` rows self-explanatory instead of looking like duplicates.

### 3. Footer polish: color + compact token counter (general improvement)

The footer is currently almost all `dim` text. Rework `RichReporter.footer()`
(`src/nextchap/cli/ui.py:132`) so each segment is visually distinct, using Rich styles
(not markup strings — the footer builds a `Text` object):

- `spend` — label dim, value **bold yellow**; when `max_cost` is set, render as
  `spent / limit` with the fraction colored: **green** under 50%, **yellow** 50–80%,
  **red** above 80% of budget.
- `cache` — rate colored by value: **dim** under 10%, **cyan** 10–50%, **green** above 50%.
- **Token counter** — replace the verbose `tokens 46,555→12,824` with a compact,
  styled counter:

  ```
  󰪞 ↑41k ↓15k
  ```

  - `↑` = prompt/input tokens (green), `↓` = completion/output tokens (magenta);
  - abbreviate: `< 1,000` → exact number, otherwise `41.2k` / `1.3M` (one decimal);
  - keep the nerd-font glyph `󰪞` (U+F0CA9, gauge icon) as the segment prefix, but see
    the fallback note below.
- **Nerd Font caveat:** `󰪞` only renders if the user's terminal font supports Nerd
  Fonts. Add a `footer_icon` setting (default `true`) under `[ui]` in the settings
  (`src/nextchap/config.py`); when `false` (or via `NEXTCHAP_UI__FOOTER_ICON=false`),
  fall back to plain labels (`in`/`out` or the current `tokens` wording). Never guess
  font support from the terminal — let the user opt out.
- Separator between segments: dim `·` or `│` instead of double spaces.
- Keep everything on one line and short enough not to wrap on an 80-col terminal.
- The per-role model display from section 1 uses the role name as its color key
  (e.g. `analyst` cyan, `writer` magenta, `planner` yellow, `critic` blue).

Example target rendering (budget set, cache warm):

```
 󰪞 ↑41.2k ↓12.8k · cache 19% · spend $0.0132/$0.05 · analyst deepseek/deepseek-v4.1-flash-1017 (15)
```

### 4. (Optional, cheap) Post-run reconciliation
- After import/generate completes, print a compact actual-usage table grouped by
  role → model (same data as `by_model`/`by_role`), so users can compare estimate vs
  actual. The usage log (`meter.bind`) already persists `CallRecord`s if needed.

### 5. Documentation note
- One line in the README (or `docs/`) stating: **import uses only the `analyst` model;
  writer/planner/critic are generation-time roles.** This prevents the same
  misreading in future.

## Tasks

1. `src/nextchap/llm/cost.py`: add `by_model()` aggregation to `CostMeter`
   (keep `by_role()` unchanged; both are cheap in-memory folds over `records`).
2. `src/nextchap/cli/ui.py::footer()`: replace the static `writer <model>` segment with
   per-role model/call counts from the meter (zero-call roles omitted; static fallback
   when no records). Do not import Rich into the engine — footer changes stay in
   `cli/ui.py` (Reporter protocol boundary, per AGENTS.md).
3. `src/nextchap/pipeline/estimate.py`: add `step` field to `EstimateRow`; populate it in
   `_row(...)` calls in both `estimate_import` and `estimate_generation`.
4. `src/nextchap/cli/app.py::_estimate_table`: add the step column to the table.
5. `src/nextchap/cli/ui.py` footer polish: add the `fmt_tokens()` helper (abbreviate to
   k/M, one decimal), color the spend fraction and cache rate by thresholds, render the
   token counter as `󰪞 ↑in ↓out`, and switch the segment separator to dim `·`.
6. `src/nextchap/config.py`: add `[ui] footer_icon = true` (env
   `NEXTCHAP_UI__FOOTER_ICON`) with a plain-text fallback rendering in the footer.
7. README: add the import-roles note.
8. Tests (no network, use `tests/fakes.py::FakeLLM` and
   `NEXTCHAP_RETRIEVAL__EMBED_BACKEND=hash`):
   - footer rendering with a meter containing mixed-role records (e.g. analyst + writer)
     and a meter with no records (static fallback);
   - estimate tables include step labels (assert via `test_cli.py` end-to-end import
     dry-run output);
   - `by_model()` totals match summing `records`;
   - `fmt_tokens()`: 999 → `999`, 41,200 → `41.2k`, 1,300,000 → `1.3M`;
   - footer with `footer_icon=false` renders no `󰪞` glyph and still shows counts;
   - footer spend colors switch at the 50%/80% budget thresholds.
9. Update/extend `tests/test_cli.py` per AGENTS.md ("New CLI command → add end-to-end
   case" — here it's a rendering change, so extend existing cases).

## Acceptance criteria

- [x] During `nextchap import`, the footer no longer shows `writer <model>` unless
      writer-role calls actually occurred in the run.
- [x] The estimate table has a step/phase column and every row is labeled.
- [x] The footer token counter reads like `󰪞 ↑41k ↓15k` (green up / magenta down,
      k/M-abbreviated), and falls back to plain text when `ui.footer_icon=false`.
- [x] Footer spend is green/yellow/red by budget fraction, cache rate dim/cyan/green
      by percentage; every footer segment is visually distinguishable.
- [x] `uv run pytest` (~90 tests, no network) passes.
- [x] `uv run ruff check . && uv run ruff format --check . && uv run mypy` all pass
      (mypy is `--strict`).

## Out of scope / do NOT do

- **Do not switch chapter/arc summaries, bible, or world-state extraction to the writer
  (haiku) model.** The premise "haiku does the summaries" was a misread of the footer;
  summaries are intentionally on the `analyst` role (deepseek) per `ModelsConfig`.
- Do not change `estimate_import` cost math — it is accurate for fresh imports.
- Do not model fallback/retry costs in the estimate (document as known approximation
  instead, if touched at all).
- Do not change journal/commit flow, `CostMeter.check_budget()`, or the usage-log format.
