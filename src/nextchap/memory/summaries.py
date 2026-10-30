"""Chapter summaries, arc summaries and the master bible."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from nextchap.ingest.master_book import Chapter
from nextchap.llm.base import LLM
from nextchap.memory.world_state import WorldState
from nextchap.prompts import render
from nextchap.retrieval.context import fit_text
from nextchap.storage.atomic import atomic_write_json, read_json

SUMMARY_INPUT_TOKENS = 12_000


@dataclass
class ChapterSummary:
    number: int
    title: str
    summary: str
    content_hash: str = ""


@dataclass
class ArcSummary:
    index: int  # 0-based
    start: int
    end: int
    summary: str


class SummaryStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> dict[int, ChapterSummary]:
        rows: list[dict[str, Any]] = read_json(self.path, default=[]) or []
        out: dict[int, ChapterSummary] = {}
        for r in rows:
            if not isinstance(r, dict) or "number" not in r:
                continue
            out[int(r["number"])] = ChapterSummary(
                number=int(r["number"]),
                title=str(r.get("title", "")),
                summary=str(r.get("summary", "")),
                content_hash=str(r.get("content_hash", "")),
            )
        return out

    def save(self, summaries: dict[int, ChapterSummary]) -> None:
        atomic_write_json(self.path, [asdict(summaries[k]) for k in sorted(summaries)])

    def put(self, summary: ChapterSummary) -> None:
        data = self.load()
        data[summary.number] = summary
        self.save(data)


class ArcStore:
    def __init__(self, path: Path, arc_size: int) -> None:
        self.path = path
        self.arc_size = arc_size

    def load(self) -> list[ArcSummary]:
        raw = read_json(self.path, default=[]) or []
        arcs: list[ArcSummary] = []
        for i, item in enumerate(raw if isinstance(raw, list) else []):
            if isinstance(item, str):  # legacy: list of strings, one per arc_size chapters
                arcs.append(ArcSummary(i, i * self.arc_size + 1, (i + 1) * self.arc_size, item))
            elif isinstance(item, dict) and "summary" in item:
                arcs.append(
                    ArcSummary(
                        int(item.get("index", i)),
                        int(item.get("start", 0)),
                        int(item.get("end", 0)),
                        str(item["summary"]),
                    )
                )
        return arcs

    def save(self, arcs: list[ArcSummary]) -> None:
        atomic_write_json(self.path, [asdict(a) for a in sorted(arcs, key=lambda a: a.index)])


def summarize_chapter(llm: LLM, ch: Chapter) -> ChapterSummary:
    comp = llm.text(
        "analyst",
        render("analyst_system"),
        render(
            "summarize_chapter",
            number=ch.number,
            title=ch.title,
            content=fit_text(ch.content, SUMMARY_INPUT_TOKENS, keep="head"),
        ),
        label=f"summary:ch{ch.number}",
        max_tokens=600,
    )
    return ChapterSummary(ch.number, ch.title, comp.text, ch.content_hash)


def summarize_arc(llm: LLM, index: int, items: list[ChapterSummary]) -> ArcSummary:
    block = "\n\n".join(f"Chapter {s.number}: {s.summary}" for s in items if s.summary)
    comp = llm.text(
        "analyst",
        render("analyst_system"),
        render(
            "summarize_arc",
            arc_number=index + 1,
            start=items[0].number,
            end=items[-1].number,
            summaries=block,
        ),
        label=f"arc:{index + 1}",
        max_tokens=1200,
    )
    return ArcSummary(index, items[0].number, items[-1].number, comp.text)


def stale_arcs(arcs: list[ArcSummary], summaries: list[ChapterSummary], arc_size: int) -> list[int]:
    """Arc indexes that are missing, or were partial and are now complete."""
    by_index = {a.index: a for a in arcs}
    out = []
    for i in range(0, len(summaries), arc_size):
        idx = i // arc_size
        chunk = summaries[i : i + arc_size]
        arc = by_index.get(idx)
        if arc is None or (len(chunk) == arc_size and arc.end != chunk[-1].number):
            out.append(idx)
    return out


def build_bible(llm: LLM, state: WorldState, arcs: list[ArcSummary]) -> str:
    recent = "\n\n".join(
        f"Arc {a.index + 1} (ch. {a.start}-{a.end}):\n{a.summary}" for a in arcs[-3:]
    )
    comp = llm.text(
        "analyst",
        render("analyst_system"),
        render("bible", world_state=state.compact(), arcs=recent or "None yet."),
        label="bible",
        max_tokens=3000,
    )
    return comp.text
