# Ticket: Footer polish — colored token counter, budget bar, and live model status

**Status:** done (branch `feat/estimate-ui-polish`)
**Priority:** medium
**Type:** UX improvement (display/styling only)
**Risk:** low — display-only changes, no pipeline or cost-logic changes

---

## Context (from investigation in ticket 0002)

The footer (`src/nextchap/cli/ui.py::RichReporter.footer()` ~line 132) currently:
1.  Hard-codes the `writer` model as a static readout (`writer anthropic/claude-haiku-5.5`) even when it's not doing any work (leading to confusion about which model is active).
2.  Displays the token counts in a verbose way (`tokens 46,555→12,824`) without visual differentiation or compression (e.g., `41.2k`).
3.  Lacks visual emphasis, almost entirely using the `dim` style.

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

### 2. Footer polish: color + compact token counter (general improvement)

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

### 3. (Optional, cheap) Post-run reconciliation
- Moved to ticket `0002-import-cost-role-transparency.md`.

## Tasks

1. `src/nextchap/llm/cost.py`: add `by_model()` aggregation to `CostMeter`
   (keep `by_role()` unchanged; both are cheap in-memory folds over `records`).
2. `src/nextchap/cli/ui.py::footer()`: replace the static `writer <model>` segment with
   per-role model/call counts from the meter (zero-call roles omitted; static fallback
   when no records). Do not import Rich into the engine — footer changes stay in
   `cli/ui.py` (Reporter protocol boundary, per AGENTS.md).
3. `src/nextchap/cli/ui.py` footer polish: add the `fmt_tokens()` helper (abbreviate to
   k/M, one decimal), color the spend fraction and cache rate by thresholds, render the
   token counter as `󰪞 ↑in ↓out`, and switch the segment separator to dim `·`.
4. `src/nextchap/config.py`: add `[ui] footer_icon = true` (env
   `NEXTCHAP_UI__FOOTER_ICON`) with a plain-text fallback rendering in the footer.
5. Tests (no network, use `tests/fakes.py::FakeLLM` and
   `NEXTCHAP_RETRIEVAL__EMBED_BACKEND=hash`):
   - footer rendering with a meter containing mixed-role records (e.g. analyst + writer)
     and a meter with no records (static fallback);
   - `by_model()` totals match summing `records`;
   - `fmt_tokens()`: 999 → `999`, 41,200 → `41.2k`, 1,300,000 → `1.3M`;
   - footer with `footer_icon=false` renders no `󰪞` glyph and still shows counts;
   - footer spend colors switch at the 50%/80% budget thresholds.
6. Update/extend `tests/test_cli.py` per AGENTS.md ("New CLI command → add end-to-end
   case" — here it's a rendering change, so extend existing cases).

## Acceptance criteria

- [x] During `nextchap import`, the footer no longer shows `writer <model>` unless
      writer-role calls actually occurred in the run.
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
- Do not modify the estimate table's step-column or cost-logic display; that is handled
  in ticket `0002-import-cost-role-transparency.md`.
- Do not change journal/commit flow, `CostMeter.check_budget()`, or the usage-log format.
