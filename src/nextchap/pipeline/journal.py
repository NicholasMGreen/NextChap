"""Per-chapter progress journal: every completed step is durable, so a crash resumes."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nextchap.prompts import PROMPT_VERSION
from nextchap.storage.atomic import atomic_write_json, atomic_write_text, read_json
from nextchap.storage.paths import BookPaths

STEPS = ("started", "planned", "written", "reviewed", "analyzed", "committing", "committed")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


@dataclass
class JournalEntry:
    chapter: int
    step: str = "started"
    prompt_version: str = PROMPT_VERSION
    updated_at: str = field(default_factory=_now)

    def reached(self, step: str) -> bool:
        return STEPS.index(self.step) >= STEPS.index(step)


class Journal:
    def __init__(self, paths: BookPaths) -> None:
        self.paths = paths

    def load(self) -> JournalEntry | None:
        data = read_json(self.paths.journal, default=None)
        if not data:
            return None
        return JournalEntry(
            **{k: v for k, v in data.items() if k in JournalEntry.__dataclass_fields__}
        )

    def start(self, chapter: int) -> JournalEntry:
        entry = JournalEntry(chapter)
        self.staging(chapter).mkdir(parents=True, exist_ok=True)
        self._save(entry)
        return entry

    def advance(self, entry: JournalEntry, step: str) -> None:
        entry.step = step
        entry.updated_at = _now()
        self._save(entry)

    def finish(self, entry: JournalEntry) -> None:
        shutil.rmtree(self.staging(entry.chapter), ignore_errors=True)
        self.paths.journal.unlink(missing_ok=True)

    def staging(self, chapter: int) -> Path:
        return self.paths.staging_for(chapter)

    def put(self, chapter: int, name: str, data: Any) -> None:
        path = self.staging(chapter) / name
        if isinstance(data, str):
            atomic_write_text(path, data)
        else:
            atomic_write_json(path, data)

    def get_text(self, chapter: int, name: str) -> str | None:
        path = self.staging(chapter) / name
        return path.read_text(encoding="utf-8") if path.exists() else None

    def get_json(self, chapter: int, name: str) -> Any:
        path = self.staging(chapter) / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _save(self, entry: JournalEntry) -> None:
        atomic_write_json(self.paths.journal, asdict(entry))
