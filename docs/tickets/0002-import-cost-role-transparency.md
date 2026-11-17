# Ticket: Comprehensive import cost estimate & role breakdown

**Status:** open
**Priority:** medium
**Type:** bug (misleading UI/logic) + UX improvement
**Risk:** low — display-only changes, no pipeline or cost-logic changes

---

## Symptom (user report)

During `nextchap import`, the pre-flight estimate table shows only `analyst` /
`deepseek/deepseek-v4.1-flash-1017` rows. The user was concerned that the estimate
wasn't comprehensive and didn't show roles like `writer` (Haiku) or other stages.

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

## Investigation findings (Context — do not re-investigate)

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

### 1. Estimate table: label each row with the pipeline step it covers
- Add a `step: str` field to `EstimateRow` (`pipeline/estimate.py`) and set it for every
  row in `estimate_import` and `estimate_generation`:
  - `"chapter summaries"`, `"arc summaries"`, `"master bible"`, `"world state"`, etc.
- Update `_estimate_table` in `cli/app.py` to include the step column.
- This makes the 4 `analyst` rows self-explanatory instead of looking like duplicates.

### 2. Post-run reconciliation
- After import/generate completes, print a compact actual-usage table grouped by
  role → model (same data as `by_model`/`by_role`), so users can compare estimate vs
  actual. The usage log (`meter.bind`) already persists `CallRecord`s if needed.

### 3. Documentation note
- One line in the README (or `docs/`) stating: **import uses only the `analyst` model;
  writer/planner/critic are generation-time roles.** This prevents the same
  misreading in future.

## Tasks

1. `src/nextchap/pipeline/estimate.py`: add `step` field to `EstimateRow`; populate it in
   `_row(...)` calls in both `estimate_import` and `estimate_generation`.
2. `src/nextchap/cli/app.py::_estimate_table`: add the step column to the table.
3. README: add the import-roles note.
4. Tests (no network, use `tests/fakes.py::FakeLLM` and
   `NEXTCHAP_RETRIEVAL__EMBED_BACKEND=hash`):
   - estimate tables include step labels (assert via `test_cli.py` end-to-end import
     dry-run output);
5. Update/extend `tests/test_cli.py` per AGENTS.md ("New CLI command → add end-to-end
   case" — here it's a rendering change, so extend existing cases).

## Acceptance criteria

- [ ] The estimate table has a step/phase column and every row is labeled.
- [ ] `uv run pytest` (~90 tests, no network) passes.
- [ ] `uv run ruff check . && uv run ruff format --check . && uv run mypy` all pass
      (mypy is `--strict`).

## Out of scope / do NOT do

- **Do not switch chapter/arc summaries, bible, or world-state extraction to the writer
  (haiku) model.** The premise "haiku does the summaries" was a misread of the footer;
  summaries are intentionally on the `analyst` role (deepseek) per `ModelsConfig`.
- Do not change `estimate_import` cost math — it is accurate for fresh imports.
- Do not model fallback/retry costs in the estimate (document as known approximation
  instead, if touched at all).
- Do not change journal/commit flow, `CostMeter.check_budget()`, or the usage-log format.
- UI footer styling/styling logic is handled in ticket `0001-import-cost-transparency.md`.
