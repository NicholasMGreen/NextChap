from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextchap.memory.world_state import (
    Character,
    CharacterPatch,
    NamedEntry,
    PlotThread,
    WorldState,
    WorldStateDelta,
    WorldStateStore,
    apply_delta,
    migrate_legacy,
)


def delta(**kw: object) -> WorldStateDelta:
    base: dict[str, object] = dict(
        upsert_characters=[],
        new_threads=[],
        resolved_thread_ids=[],
        new_world_rules=[],
        upsert_locations=[],
        upsert_factions=[],
        new_key_items=[],
        current_arc=None,
        story_tone=None,
        narrative_pov=None,
        last_chapter_hook="It ends.",
    )
    base.update(kw)
    return WorldStateDelta.model_validate(base)


def patch(name: str, id: str | None = None, **fields: str) -> CharacterPatch:
    keys = ("role", "status", "location", "power_level", "goals", "relationships", "notes")
    return CharacterPatch(id=id, name=name, **{k: fields.get(k) for k in keys})


@pytest.fixture
def state() -> WorldState:
    return WorldState(
        characters=[Character(id="char_lin", name="Lin Feng", status="injured")],
        threads=[PlotThread(id="t001", description="Find the traitor")],
        chapter_count=10,
    )


def test_apply_delta_does_not_mutate_input(state: WorldState) -> None:
    before = state.model_dump()
    new = apply_delta(
        state, delta(upsert_characters=[patch("Lin Feng", "char_lin", status="healed")]), 11
    )
    assert state.model_dump() == before
    assert new.character("char_lin").status == "healed"  # type: ignore[union-attr]
    assert new.chapter_count == 11


def test_resolve_by_id_and_add_threads(state: WorldState) -> None:
    new = apply_delta(
        state, delta(resolved_thread_ids=["t001", "t999"], new_threads=["Who sent the letter?"]), 11
    )
    assert new.threads[0].status == "resolved" and new.threads[0].resolved_in == 11
    assert new.threads[1].id == "t002" and new.threads[1].introduced_in == 11


def test_new_character_and_name_match(state: WorldState) -> None:
    new = apply_delta(
        state,
        delta(upsert_characters=[patch("lin feng", location="Peak"), patch("Mei", role="ally")]),
        11,
    )
    assert len(new.characters) == 2
    assert new.character("Lin Feng").location == "Peak"  # type: ignore[union-attr]
    assert new.character("Mei").id == "char_mei"  # type: ignore[union-attr]


def test_rejects_missing_hook_and_old_chapter(state: WorldState) -> None:
    from nextchap.memory.world_state import DeltaRejected

    with pytest.raises(DeltaRejected):
        apply_delta(state, delta(last_chapter_hook="  "), 11)
    with pytest.raises(DeltaRejected):
        apply_delta(state, delta(), 9)


def test_named_entries_merge(state: WorldState) -> None:
    new = apply_delta(
        state, delta(upsert_locations=[NamedEntry(name="Peak", description="cold")]), 11
    )
    newer = apply_delta(
        new, delta(upsert_locations=[NamedEntry(name="peak", description="snowy")]), 12
    )
    assert [(e.name, e.description) for e in newer.locations] == [("Peak", "snowy")]


def test_migrate_legacy_v0() -> None:
    legacy = {
        "characters": {
            "Lin Feng": {"role": "hero", "status": "alive", "power_level": {"realm": 3}},
            "Old Man": "mysterious",
        },
        "active_plot_threads": ["A", "B"],
        "resolved_plot_threads": ["C"],
        "world_rules": ["r1"],
        "locations": {"Peak": "cold"},
        "factions": {"Sect": "strong"},
        "key_items": ["sword"],
        "current_arc": "arc",
        "story_tone": "dark",
        "narrative_pov": "3rd",
        "last_chapter_hook": "hook",
        "chapter_count": 42,
    }
    s = migrate_legacy(legacy)
    assert s.chapter_count == 42
    assert s.character("Lin Feng").power_level == '{"realm": 3}'  # type: ignore[union-attr]
    assert s.character("Old Man").notes == "mysterious"  # type: ignore[union-attr]
    assert [t.id for t in s.active_threads] == ["t001", "t002"]
    assert s.threads[2].status == "resolved"
    assert s.locations[0].name == "Peak"


def test_store_migrates_and_snapshots(tmp_path: Path) -> None:
    path = tmp_path / "world_state.json"
    path.write_text(
        json.dumps({"characters": {}, "active_plot_threads": ["x"], "chapter_count": 3})
    )
    store = WorldStateStore(path, tmp_path / "hist")
    s = store.load()
    assert s.schema_version == 1
    assert (tmp_path / "world_state.v0.json").exists()
    store.save(s.model_copy(update={"chapter_count": 4}))
    store.save(s.model_copy(update={"chapter_count": 5}))
    assert store.snapshots() == [3, 4, 5]
    rolled = store.rollback(4)
    assert rolled.chapter_count == 4 and store.load().chapter_count == 4
    assert store.snapshots() == [3, 4]


def test_compact_lists_only_active_threads(state: WorldState) -> None:
    state.threads.append(PlotThread(id="t002", description="Old", status="resolved"))
    data = json.loads(state.compact())
    assert [t["id"] for t in data["active_threads"]] == ["t001"]
