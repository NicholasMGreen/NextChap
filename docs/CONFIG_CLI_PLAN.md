# Plan: `nextchap config` as a real settings CLI

Status: implemented on `feat/config-cli` (2026-11-05). Owner decision recorded
below; implementation notes and deviations at the end. Scope: the `config`
command, a new settings browser, and the config-file write path. Nothing in the
generation pipeline, retrieval, or book storage was touched.

## Goal

Today settings can only be changed by someone who already knows the exact
dotted key and the TOML/JSON shape of its value:

```
nextchap config set models.writer.top_p 0.95 --project
```

There is no way to *see* which keys exist, what values are allowed, what the
current value is, or *why* a value is what it is (default, project file, user
file, `.env`, environment variable). The bare `nextchap` menu has no settings
entry at all.

The overhaul makes `nextchap config` a complete, discoverable settings CLI:

1. **Browse and edit** every setting from a keyed picker — no keys to memorise.
2. **Every interactive action has a non-interactive equivalent** (`get`, `list`,
   `set`, `unset`, `keys --json`), so scripts and agents never need the picker.
3. **Show provenance** — the resolved value *and* which layer supplied it.
4. **Validate before writing**, atomically, with a one-step undo.

## Decisions (owner-confirmed)

1. **Rich only, no new dependency.** Keys, not arrow keys: a numbered/lettered
   picker with a `/`-filter line, exactly the idiom the existing `nextchap`
   menu and `fix` repair menu already use. Terminal-agnostic and testable by
   feeding stdin.
2. **The browser is a shell over the verbs** (same rule as `cli/menu.py`): it
   builds a call to `config set`/`config unset` and nothing else.
3. **One schema source.** A single descriptor registry drives validation, help
   text, allowed values, the picker, `--json` output and the tests. No second
   hand-written list of keys that can drift.
4. **Writes default to the user config**, `--project` (→ `./nextchap.toml`)
   only on request. Same as today.

## Audit of the current code (what this plan has to fix)

| Where | Today | Problem |
|---|---|---|
| `cli/commands_extra.py:349-453` | `show`, `path`, `set`, `set-key` | no `get`, `list`, `unset`, `edit`, `keys`; no discovery |
| `config_set` | `key.split(".")`, `_parse_value` (JSON then raw string) | user must know the type; `top_p` must be `0.95`, `fallbacks` must be `["a/b"]` |
| `config_set` validation | `Settings(**data)` on the layer file | pydantic message only (`errors()[0]['msg']`), no hint about the key |
| `config_set` write | `path.write_text(...)` | **violates the "every write goes through `storage/atomic.py`" invariant** |
| `config show` | resolved values, key masked | no provenance; a value that looks wrong in `show` may be an env var, and nothing says so |
| `models` command | separate read-only readout | writer/planner/analyst/critic settings live under `models.*` but you can't edit them from there |
| `cli/menu.py` | book picker → action picker | decisions 5 in `CLI_SIMPLIFICATION_PLAN.md` didn't add settings; menu always starts by picking a book |

## New CLI surface

```
nextchap config                          # interactive settings browser (this is the upgrade)
nextchap config list                     # keys that differ from defaults, with source column
nextchap config list --all               # every key
nextchap config list --section models --json
nextchap config get generation.min_words # value + source layer
nextchap config set KEY VALUE [--project]    # unchanged signature, better errors
nextchap config unset KEY [--project] [--restore]
nextchap config keys [--json]            # machine-readable schema: type, default, choices, help
nextchap config edit [--project]         # $EDITOR on the toml, validated on save
nextchap config show [--json]            # unchanged
nextchap config path                     # unchanged, plus a "which layer wins" summary
```

Backwards compatibility: the existing `config set` / `config show` / `config
path` argument shapes and success messages stay byte-identical, so
`tests/test_cli.py::test_config_set_and_show` keeps passing unchanged.

### The browser

```
NextChap settings · 42 keys · user config
 ┌──────────────────────────────────────────────────────────────────────┐
 │ #  Key                              Value              Source        │
 │ ── models                                                           │
 │ 1  models.writer.model               deepseek/deepseek…  default      │
 │ 2  models.writer.temperature         0.9                 default      │
 │ 3  models.critic.model               anthropic/claude…   project      │
 │ ── defaults                                                         │
 │ 4  defaults.chapters                 3                   user         │
 │ …                                                                    │
 └──────────────────────────────────────────────────────────────────────┘
 1-9 · /filter · [w]here nextchap.toml is · [q]uit
 Setting 1 (type /filter to narrow)
```

- Sections mirror `Settings`: `models` (per role), `defaults`, `generation`,
  `retrieval`, `ui`, `cleanup`, `network`.
- The filter line matches key *and* help text; `/review` → the three
  `defaults.review*` keys. Filter is sticky until cleared with `/`.
- Picking a key opens the right editor for its type:

| Type | Prompt |
|---|---|
| `bool` | `Value [y/n] (current: true)` |
| `int` / `float` + min/max | `Value (0-100, current: 6)`, range-checked before write |
| `Literal[...]` | numbered choices (`1 local 2 remote 3 hash`) |
| `list[float]` (e.g. `target_words`) | `1800, 2400`, comma-split, length checked |
| `list[str]` (`extra_noise_patterns`) | one per line, empty line ends |
| model id (`models.*.model`, `fallbacks`) | **numbered pick from the model registry**, plus `m` to type an id manually |
| `api_key` | refused with the existing hint (`config set-key` / `OPENROUTER_API_KEY`) |

- After a successful edit the table re-renders with the row marked `✓ changed`
  and the new source badge; invalid input prints the error inline and returns
  to the table instead of exiting (same ergonomics as the `fix` menu).
- Writing a value that an environment variable shadows prints
  `! written, but NEXTCHAP_GENERATION__MIN_WORDS=2000 still wins`.
- Non-interactive (`not app_ctx.interactive`, `--plain`, or `not sys.stdin.isatty()`)
  → print `config show` plus a one-line hint. The browser never blocks a pipe,
  CI, or a `nextchap config` call inside a script.

## Implementation

New modules:

```
src/nextchap/config_schema.py     descriptors: key, type, default, choices, min/max, help
src/nextchap/config_file.py       layer read → merge → validate → atomic write → backup
src/nextchap/cli/prompt_value.py  typed prompt: parse, validate, retry (used by the browser)
src/nextchap/cli/settings_browser.py  the picker; builds config set/unset calls only
```

`config.py` keeps `Settings` as the only source of truth for *values*;
`config_schema.py` is the only source of truth for *metadata about* values, and
is generated by walking `Settings.model_fields` (recursing into the nested
models) so a new field appears automatically. A test asserts
`{d.key for d in DESCRIPTORS} == {leaf keys of Settings}` minus a short,
explicit deny-list (`api_key`), so adding a persisted field without metadata
fails CI.

`config_file.py` API:

```python
load_layer(path: Path) -> dict[str, Any]
merged_with_value(data: dict, key: str, value: Any) -> dict      # creates tables, errors on conflicts
validate_layer(data: dict, key: str) -> Settings                  # whole-file validate, ConfigError with hint
write_layer(path: Path, data: dict) -> None                      # storage.atomic + config.toml.bak
source_of(key: str, settings: Settings) -> Literal["env", "dotenv", "project", "user", "default"]
```

`source_of` reads the two TOML files and the derived env var name
(`NEXTCHAP_` + key uppercased, `__` for dots) — no network, no model calls.

## Phases

Each phase ends with pytest, `ruff check`, `ruff format --check`, `mypy
--strict` green, and the phase is committed on its own.

**Phase 1 — schema registry.** `config_schema.py` + full-key-coverage test +
help text for every setting (short, plain English: what it changes, when to
touch it). `config keys --json` wired to it. No behaviour change.

**Phase 2 — non-interactive verbs and safe writes.** `config_file.py`;
`show`/`set`/`path` moved onto it (atomic write + backup + whole-file
validation + env-shadow warning); new `get`, `list`, `unset [--restore]`,
`edit`. Tests: valid/invalid value per type, non-table conflict, env shadow
warning, backup restored by `--restore`, `api_key` still refused, existing
`test_config_set_and_show` untouched and passing.

**Phase 3 — typed prompts.** `cli/prompt_value.py` with a per-type editor and
a retry loop; unit-tested by monkeypatching `rich.prompt.Prompt.ask` (the
existing test idiom) — bool, enum, range, list, model pick (registry stubbed
from the offline price table, never HTTP).

**Phase 4 — the browser.** `cli/settings_browser.py`, invoked when `config`
gets no subcommand and the session is interactive; `--section KEY` to open
straight into a section. Tests: filter narrows the list, picking a key edits
and re-renders, invalid value returns to the table, `q` exits 0, non-tty path
prints `show` and exits 0.

**Phase 5 — reachability.** Add a book-independent `settings` row to the bare
`nextchap` menu (it currently picks a book first — the row is offered before
the book picker and returns `["config"]`), and a `doctor` check for settings
shadowed by environment variables. Update `nextchap --help` epilogs and the
`EXAMPLES_CONFIG` block.

**Phase 6 — docs.** README "Configuration" section rewritten around the
browser + verbs table; `CHANGELOG.md` Unreleased entry; note the atomic-write
fix.

## Tests (no network, per repo rule)

- `tests/test_config_schema.py` — key coverage both directions, every
  descriptor's `choices` match the pydantic `Literal`, defaults match
  `Settings()`.
- `tests/test_config_file.py` — merge/validate/write/backup round-trips,
  atomic write monkeypatched to prove it is used, invalid layer leaves the old
  file intact.
- `tests/test_cli.py` — end-to-end additions for `get`/`list`/`unset`/`keys`,
  and the browser driven through `CliRunner(input=...)`.
- `tests/test_prompt_value.py` — per-type parsing and retry.

## Guardrails (unchanged invariants)

- No new runtime dependencies (rich + typer only). No arrow-key TUI.
- Engine, journaling, `sync()` idempotency, `Reporter`, book-ID validation: untouched.
- Every write goes through `storage/atomic.py`; UTF-8; user config is never
  written without an explicit verb or a picker edit.
- `api_key` is never written to a config file and is masked in every output.
- Tests never call the network; the model picker uses the offline price table.
- `nextchap config set` keeps its current signature and messages.

## Out of scope

- Editing per-book overrides (`book.json`) — settings stay global.
- A config REPL outside `config`; the browser exits on `q`.
- Live model registry refresh inside the browser (it uses the 24h cache and the
  offline fallback; `nextchap models --refresh` stays the way to refresh).
- Validating that a chosen model id exists at OpenRouter — a typo is caught by
  the first API call, as today.

## Risks

- **Schema drift** is the only real one; the both-directions coverage test is
  what keeps it honest. If a field must not be user-editable, it goes in the
  deny-list with a reason, not silently.
- **Provenance guessing:** `source_of` re-derives env var names rather than
  asking pydantic-settings. A unit test pins the derivation against
  `Settings(env_nested_delimiter="__")` behaviour for nested keys.
- Rollback is cheap: the browser and `config_file.py` are additive; Phase 2 is
  the only phase that changes existing behaviour, and it is a single commit.

## Implementation notes (what actually shipped)

Built in parallel git worktrees, one phase per branch, merged into
`feat/config-cli` in four steps. Final: 278 tests (was 226), ruff and
mypy --strict clean.

Modules, as planned: `config_schema.py` (63 editable settings + `api_key`),
`config_file.py`, `cli/prompt_value.py`, `cli/settings_browser.py`. The `config`
app in `cli/commands_extra.py` gained `get`, `list`, `keys`, `unset`, `edit`, a
callback for the browser, and a `doctor` row; `cli/menu.py` gained the settings
action.

Deviations worth knowing:

1. **Defaults come from `Settings.model_construct()`, not `Settings()`.** A live
   `Settings()` reads the developer's own `config.toml` and environment, so the
   registry would advertise personal values as "the default". Effective values
   are read from a live `Settings()` at render time, which is what makes the
   changed-from-default marker meaningful.
2. **`SettingSpecProtocol` members are read-only properties.** As plain
   attributes they were invariant, so a frozen `SettingSpec` (whose `kind` is a
   `Literal`) could never satisfy the protocol. Caught at integration, not by
   the unit tests.
3. **`data_dir` is reachable but never counts as "changed".** `--data-dir` and
   `NEXTCHAP_HOME` are app-level overrides, not config-layer values, so
   `config list` and the browser's non-interactive summary exclude it (it still
   appears in the browser's "Data location" group and in `config get`).
   Consequence: `config get data_dir` can show a value that differs from the
   default while `source` reads `default`.
4. **`config set` now rejects unknown keys** (previously any dotted key was
   written). This is the one deliberate behaviour change in the surface;
   `config set defaults.chapters 0` and `config set api_key …` keep their old
   refusals.
5. **`tests/conftest.py` now patches both `user_config_path` bindings**
   (`nextchap.config` and `nextchap.config_file`). Patching only the first let
   the layer module reach the developer's real `config.toml` inside tests.
6. `env_shadow` uses loose comparison (case and quoting only), documented in
   its docstring; it is a warning, never a validation.
