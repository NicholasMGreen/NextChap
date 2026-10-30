"""Structured story state with stable IDs, validated delta updates, and per-chapter snapshots."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nextchap.storage.atomic import atomic_write_json, atomic_write_text, read_json
from nextchap.storage.paths import slugify

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


class Character(BaseModel):
    id: str
    name: str
    role: str = ""
    status: str = ""
    location: str = ""
    power_level: str = ""
    goals: str = ""
    relationships: str = ""
    notes: str = ""


class PlotThread(BaseModel):
    id: str
    description: str
    introduced_in: int | None = None
    status: Literal["active", "resolved"] = "active"
    resolved_in: int | None = None


class NamedEntry(BaseModel):
    name: str
    description: str


class WorldState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    characters: list[Character] = Field(default_factory=list)
    threads: list[PlotThread] = Field(default_factory=list)
    world_rules: list[str] = Field(default_factory=list)
    locations: list[NamedEntry] = Field(default_factory=list)
    factions: list[NamedEntry] = Field(default_factory=list)
    key_items: list[str] = Field(default_factory=list)
    current_arc: str = ""
    story_tone: str = ""
    narrative_pov: str = ""
    last_chapter_hook: str = ""
    chapter_count: int = 0

    @property
    def active_threads(self) -> list[PlotThread]:
        return [t for t in self.threads if t.status == "active"]

    def character(self, ref: str) -> Character | None:
        key = ref.strip().casefold()
        for c in self.characters:
            if c.id == ref or c.name.casefold() == key:
                return c
        return None

    def new_character_id(self, name: str) -> str:
        base = f"char_{slugify(name)}"
        taken = {c.id for c in self.characters}
        cid, n = base, 2
        while cid in taken:
            cid, n = f"{base}_{n}", n + 1
        return cid

    def new_thread_id(self) -> str:
        nums = [int(m.group(1)) for t in self.threads if (m := re.fullmatch(r"t(\d+)", t.id))]
        return f"t{(max(nums) + 1) if nums else 1:03d}"

    def compact(self, max_characters: int = 40, max_rules: int = 30) -> str:
        """Prompt-friendly JSON: active threads only, capped lists."""
        data = {
            "characters": [
                c.model_dump(exclude_defaults=True) for c in self.characters[:max_characters]
            ],
            "active_threads": [
                {"id": t.id, "description": t.description} for t in self.active_threads
            ],
            "world_rules": self.world_rules[:max_rules],
            "locations": [e.model_dump() for e in self.locations[:25]],
            "factions": [e.model_dump() for e in self.factions[:20]],
            "key_items": self.key_items[:25],
            "current_arc": self.current_arc,
            "story_tone": self.story_tone,
            "narrative_pov": self.narrative_pov,
            "last_chapter_hook": self.last_chapter_hook,
        }
        return json.dumps(data, ensure_ascii=False, indent=1)


# ── LLM-facing schemas ────────────────────────────────────────────────────────


class CharacterDraft(BaseModel):
    name: str
    role: str
    status: str
    location: str
    power_level: str
    goals: str
    relationships: str
    notes: str


class WorldStateExtraction(BaseModel):
    """Initial extraction from chapter summaries (IDs are assigned in code)."""

    characters: list[CharacterDraft]
    active_threads: list[str]
    resolved_threads: list[str]
    world_rules: list[str]
    locations: list[NamedEntry]
    factions: list[NamedEntry]
    key_items: list[str]
    current_arc: str
    story_tone: str
    narrative_pov: str
    last_chapter_hook: str


class CharacterPatch(BaseModel):
    id: str | None = Field(description="Existing character id, or null for a new character")
    name: str
    role: str | None
    status: str | None
    location: str | None
    power_level: str | None
    goals: str | None
    relationships: str | None
    notes: str | None


class WorldStateDelta(BaseModel):
    upsert_characters: list[CharacterPatch]
    new_threads: list[str]
    resolved_thread_ids: list[str]
    new_world_rules: list[str]
    upsert_locations: list[NamedEntry]
    upsert_factions: list[NamedEntry]
    new_key_items: list[str]
    current_arc: str | None
    story_tone: str | None
    narrative_pov: str | None
    last_chapter_hook: str = Field(description="The ending hook of this chapter (required)")


class DeltaRejected(ValueError):
    pass


def from_extraction(x: WorldStateExtraction, chapter_count: int) -> WorldState:
    state = WorldState(chapter_count=chapter_count)
    for draft in x.characters:
        if not draft.name.strip() or state.character(draft.name):
            continue
        state.characters.append(
            Character(id=state.new_character_id(draft.name), **draft.model_dump())
        )
    for desc in x.active_threads:
        state.threads.append(PlotThread(id=state.new_thread_id(), description=desc))
    for desc in x.resolved_threads:
        state.threads.append(
            PlotThread(id=state.new_thread_id(), description=desc, status="resolved")
        )
    state.world_rules = _dedupe(x.world_rules)
    state.locations = _merge_named([], x.locations)
    state.factions = _merge_named([], x.factions)
    state.key_items = _dedupe(x.key_items)
    state.current_arc = x.current_arc
    state.story_tone = x.story_tone
    state.narrative_pov = x.narrative_pov
    state.last_chapter_hook = x.last_chapter_hook
    return state


def apply_delta(state: WorldState, delta: WorldStateDelta, chapter: int) -> WorldState:
    """Return a new state; the input is never mutated. Raises DeltaRejected on bad updates."""
    if chapter < state.chapter_count:
        raise DeltaRejected(
            f"Delta for chapter {chapter} is older than state ({state.chapter_count})."
        )
    if not delta.last_chapter_hook.strip():
        raise DeltaRejected("Delta is missing last_chapter_hook.")

    new = state.model_copy(deep=True)
    for patch in delta.upsert_characters:
        target = (new.character(patch.id) if patch.id else None) or new.character(patch.name)
        fields = {k: v for k, v in patch.model_dump(exclude={"id", "name"}).items() if v}
        if target is None:
            if not patch.name.strip():
                continue
            new.characters.append(
                Character(id=new.new_character_id(patch.name), name=patch.name, **fields)
            )
        else:
            for k, v in fields.items():
                setattr(target, k, v)

    known = {t.id: t for t in new.threads}
    for tid in delta.resolved_thread_ids:
        thread = known.get(tid.strip())
        if thread is None:
            log.warning("Delta resolved unknown thread id %r; ignored.", tid)
        elif thread.status == "active":
            thread.status, thread.resolved_in = "resolved", chapter
    existing_desc = {t.description.casefold() for t in new.threads}
    for desc in delta.new_threads:
        if desc.strip() and desc.casefold() not in existing_desc:
            new.threads.append(
                PlotThread(id=new.new_thread_id(), description=desc, introduced_in=chapter)
            )
            existing_desc.add(desc.casefold())

    new.world_rules = _dedupe([*new.world_rules, *delta.new_world_rules])
    new.key_items = _dedupe([*new.key_items, *delta.new_key_items])
    new.locations = _merge_named(new.locations, delta.upsert_locations)
    new.factions = _merge_named(new.factions, delta.upsert_factions)
    for attr in ("current_arc", "story_tone", "narrative_pov"):
        value = getattr(delta, attr)
        if value and value.strip():
            setattr(new, attr, value.strip())
    new.last_chapter_hook = delta.last_chapter_hook.strip()
    new.chapter_count = chapter

    if len(new.characters) < len(state.characters):
        raise DeltaRejected("Delta would remove characters.")
    return new


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = item.strip().casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip())
    return out


def _merge_named(base: list[NamedEntry], updates: list[NamedEntry]) -> list[NamedEntry]:
    out = [e.model_copy() for e in base]
    index = {e.name.casefold(): e for e in out}
    for u in updates:
        if not u.name.strip():
            continue
        hit = index.get(u.name.casefold())
        if hit is None:
            entry = NamedEntry(name=u.name.strip(), description=u.description.strip())
            out.append(entry)
            index[entry.name.casefold()] = entry
        elif u.description.strip():
            hit.description = u.description.strip()
    return out


# ── Legacy (v0) migration ─────────────────────────────────────────────────────


def migrate_legacy(data: dict[str, Any]) -> WorldState:
    """Convert the original dataclass dump (dict-keyed characters, string threads)."""
    state = WorldState(chapter_count=int(data.get("chapter_count") or 0))
    chars = data.get("characters") or {}
    if isinstance(chars, dict):
        for name, info in chars.items():
            fields: dict[str, str] = {}
            if isinstance(info, dict):
                for k in (
                    "role",
                    "status",
                    "location",
                    "power_level",
                    "goals",
                    "relationships",
                    "notes",
                ):
                    v = info.get(k)
                    if v:
                        fields[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            elif info:
                fields["notes"] = str(info)
            state.characters.append(
                Character(id=state.new_character_id(str(name)), name=str(name), **fields)
            )
    for desc in data.get("active_plot_threads") or []:
        state.threads.append(PlotThread(id=state.new_thread_id(), description=str(desc)))
    for desc in data.get("resolved_plot_threads") or []:
        state.threads.append(
            PlotThread(id=state.new_thread_id(), description=str(desc), status="resolved")
        )

    def named(obj: Any) -> list[NamedEntry]:
        if isinstance(obj, dict):
            return [NamedEntry(name=str(k), description=str(v)) for k, v in obj.items()]
        if isinstance(obj, list):
            return [NamedEntry(name=str(v), description="") for v in obj]
        return []

    state.locations = named(data.get("locations"))
    state.factions = named(data.get("factions"))
    state.world_rules = [str(r) for r in data.get("world_rules") or []]
    state.key_items = [str(r) for r in data.get("key_items") or []]
    for attr in ("current_arc", "story_tone", "narrative_pov", "last_chapter_hook"):
        setattr(state, attr, str(data.get(attr) or ""))
    return state


# ── Persistence ───────────────────────────────────────────────────────────────


class WorldStateStore:
    def __init__(self, path: Path, history_dir: Path) -> None:
        self.path = path
        self.history_dir = history_dir

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> WorldState:
        data = read_json(self.path, default=None)
        if data is None:
            return WorldState()
        if "schema_version" not in data:
            state = migrate_legacy(data)
            backup = self.path.with_suffix(".v0.json")
            if not backup.exists():
                atomic_write_text(backup, self.path.read_text(encoding="utf-8"))
            self.save(state)
            return state
        return WorldState.model_validate(data)

    def save(self, state: WorldState) -> None:
        payload = state.model_dump(mode="json")
        atomic_write_json(self.path, payload)
        atomic_write_json(self.history_dir / f"ch{state.chapter_count:05d}.json", payload)

    def snapshots(self) -> list[int]:
        if not self.history_dir.exists():
            return []
        out = []
        for p in self.history_dir.glob("ch*.json"):
            m = re.fullmatch(r"ch(\d+)\.json", p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def load_snapshot(self, chapter: int) -> WorldState:
        path = self.history_dir / f"ch{chapter:05d}.json"
        return WorldState.model_validate(read_json(path))

    def rollback(self, chapter: int) -> WorldState:
        state = self.load_snapshot(chapter)
        atomic_write_json(self.path, state.model_dump(mode="json"))
        for n in self.snapshots():
            if n > chapter:
                (self.history_dir / f"ch{n:05d}.json").unlink(missing_ok=True)
        return state
