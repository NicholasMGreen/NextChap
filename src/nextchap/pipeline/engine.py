"""Orchestration: import, sync (self-healing), and journaled chapter generation.

Commit order for a generated chapter (each write atomic, each step idempotent):
master_book.md -> world_state.json -> vector index -> chapter_summaries.json -> book.json.
A crash anywhere leaves a journal entry; the next run finishes or resumes it, so no step
is applied twice.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from nextchap.config import Settings
from nextchap.errors import (
    BudgetExceeded,
    FatalLLMError,
    GenerationAborted,
    LLMError,
    ParseError,
)
from nextchap.ingest import master_book
from nextchap.ingest.epub import EpubMeta, write_epub
from nextchap.ingest.master_book import Chapter, MasterBook
from nextchap.llm.base import LLM
from nextchap.llm.cost import CostMeter
from nextchap.memory.summaries import (
    ChapterSummary,
    build_bible,
    stale_arcs,
    summarize_arc,
    summarize_chapter,
)
from nextchap.memory.world_state import (
    DeltaRejected,
    WorldState,
    WorldStateDelta,
    WorldStateExtraction,
    apply_delta,
    from_extraction,
)
from nextchap.pipeline.book import Book, BookMeta
from nextchap.pipeline.critic import CriticReport, critique, local_checks
from nextchap.pipeline.humanize import humanize_findings
from nextchap.pipeline.journal import Journal, JournalEntry
from nextchap.pipeline.plan import PlanDoc, parse_plan, render_plan
from nextchap.pipeline.planner import plan_chapter
from nextchap.pipeline.reporter import NullReporter, Reporter, ReviewHooks
from nextchap.pipeline.writer import (
    extend_chapter,
    join_scenes,
    rewrite_scene,
    word_count,
    write_scene,
)
from nextchap.prompts import raw, render
from nextchap.retrieval.context import ContextBudget, fit_items, fit_text
from nextchap.retrieval.store import VectorStore
from nextchap.storage.atomic import atomic_write_text

SUMMARY_SAVE_EVERY = 10
INDEX_BATCH = 10
WORLD_INIT_SUMMARIES = 60


class Engine:
    def __init__(
        self,
        book: Book,
        llm: LLM,
        meter: CostMeter,
        store_factory: Callable[[], VectorStore],
        reporter: Reporter | None = None,
        review: ReviewHooks | None = None,
    ) -> None:
        self.book = book
        self.settings: Settings = book.settings
        self.cfg = self.settings.generation
        self.llm = llm
        self.meter = meter
        self.ui: Reporter = reporter or NullReporter()
        self.review = review
        self.journal = Journal(book.paths)
        self.budget = ContextBudget.from_total(self.cfg.max_prompt_tokens)
        self.stop_requested = False
        self._store_factory = store_factory
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def _checkpoint(self) -> None:
        if self.stop_requested:
            raise GenerationAborted("Stopped at a safe point.")

    # ── import ────────────────────────────────────────────────────────────────

    def import_chapters(
        self, chapters: list[Chapter], meta: BookMeta, template: Path | None
    ) -> None:
        paths = self.book.paths
        paths.root.mkdir(parents=True, exist_ok=True)
        if template is not None:
            shutil.copy2(template, paths.template_epub)
        master_book.save(paths.master_book, MasterBook({"source": meta.source_name}, chapters))
        meta.source_chapters = len(chapters)
        self.book.save_meta(meta)
        self.book.style()
        self.sync()

    # ── sync: bring memory in line with master_book.md ────────────────────────

    def sync(self) -> None:
        self._recover()
        mb = self.book.master()
        if not mb.chapters:
            raise ParseError(f"{self.book.paths.master_book} contains no chapters.")
        self._sync_summaries(mb)
        self._sync_world(mb)
        self._sync_arcs()
        if not self.book.bible().strip():
            self.ui.step("Writing the master bible")
            self.book.save_bible(
                build_bible(self.llm, self.book.world.load(), self.book.arcs.load())
            )
        self._sync_index(mb)

    def _recover(self) -> None:
        entry = self.journal.load()
        if entry is not None and entry.reached("committing"):
            self.ui.info(f"Finishing the interrupted commit of chapter {entry.chapter}.")
            self._commit(entry)
            self._maintain(entry)

    def _sync_summaries(self, mb: MasterBook) -> None:
        summaries = self.book.summaries.load()
        dirty = False
        for ch in mb.chapters:
            s = summaries.get(ch.number)
            if s is not None and not s.content_hash:  # legacy summary: adopt current text
                s.content_hash, dirty = ch.content_hash, True
        todo = [
            c
            for c in mb.chapters
            if c.number not in summaries or summaries[c.number].content_hash != c.content_hash
        ]
        edited = [c.number for c in todo if c.number in summaries]
        if edited:
            self.ui.warn(
                f"{len(edited)} chapter(s) changed since they were summarized ({_short(edited)}); "
                "re-summarizing and re-indexing. The world state is not re-derived; check it with "
                "`nextchap state show`."
            )
        if todo:
            self.ui.phase("Summarizing chapters", len(todo))
            for i, ch in enumerate(todo, 1):
                self._checkpoint()
                summaries[ch.number] = summarize_chapter(self.llm, ch)
                self.ui.advance(detail=f"ch {ch.number}")
                if i % SUMMARY_SAVE_EVERY == 0:
                    self.book.summaries.save(summaries)
            self.ui.end_phase()
            dirty = True
        stale = sorted(set(summaries) - {c.number for c in mb.chapters})
        for n in stale:
            del summaries[n]
            dirty = True
        if dirty:
            self.book.summaries.save(summaries)

    def _ordered_summaries(self) -> list[ChapterSummary]:
        data = self.book.summaries.load()
        return [data[k] for k in sorted(data)]

    def _sync_world(self, mb: MasterBook) -> None:
        if not self.book.world.exists():
            self.ui.step("Extracting the world state")
            items = [f"Chapter {s.number}: {s.summary}" for s in self._ordered_summaries()]
            block = "\n\n".join(fit_items(items[-WORLD_INIT_SUMMARIES:], self.budget.recent_prose))
            extraction = self.llm.structured(
                "analyst",
                render("analyst_system"),
                render("world_init", summaries=block),
                WorldStateExtraction,
                label="world:init",
            )
            self.book.world.save(from_extraction(extraction, mb.last_number))
            return
        state = self.book.world.load()
        if state.chapter_count > mb.last_number:
            self.ui.warn(
                f"world_state.json says chapter {state.chapter_count} but master_book.md ends at "
                f"{mb.last_number}; resetting the counter."
            )
            state = state.model_copy(update={"chapter_count": mb.last_number})
            self.book.world.save(state)
        new = [c for c in mb.chapters if c.number > state.chapter_count]
        if new:
            self.ui.phase("Updating world state for new chapters", len(new))
            for ch in new:
                self._checkpoint()
                state = self._next_state(state, ch, fallback_hook=state.last_chapter_hook)
                self.book.world.save(state)
                self.ui.advance(detail=f"ch {ch.number}")
            self.ui.end_phase()

    def _sync_arcs(self) -> None:
        size = self.cfg.arc_size
        ordered = self._ordered_summaries()
        arcs = self.book.arcs.load()
        todo = stale_arcs(arcs, ordered, size)
        if not todo:
            return
        self.ui.phase("Summarizing arcs", len(todo))
        by_index = {a.index: a for a in arcs}
        for idx in todo:
            self._checkpoint()
            by_index[idx] = summarize_arc(self.llm, idx, ordered[idx * size : (idx + 1) * size])
            self.book.arcs.save(list(by_index.values()))
            self.ui.advance(detail=f"arc {idx + 1}")
        self.ui.end_phase()

    def _sync_index(self, mb: MasterBook) -> None:
        indexed = self.store.indexed()
        want = {c.number for c in mb.chapters}
        gone = sorted(set(indexed) - want)
        if gone:
            self.store.delete_chapters(gone)
        todo = [c for c in mb.chapters if indexed.get(c.number) != c.content_hash]
        if not todo:
            return
        self.ui.phase("Indexing chapters", len(todo))
        for i in range(0, len(todo), INDEX_BATCH):
            batch = todo[i : i + INDEX_BATCH]
            self.store.upsert_chapters(batch)
            self.ui.advance(len(batch))
        self.ui.end_phase()

    def _next_state(self, state: WorldState, ch: Chapter, fallback_hook: str) -> WorldState:
        try:
            delta = self.llm.structured(
                "analyst",
                render("analyst_system"),
                render(
                    "world_delta",
                    chapter=ch.number,
                    state=state.compact(),
                    content=fit_text(ch.content, 12_000),
                ),
                WorldStateDelta,
                label=f"world:ch{ch.number}",
            )
            return apply_delta(state, delta, ch.number)
        except FatalLLMError:
            raise
        except (LLMError, DeltaRejected) as exc:
            self.ui.warn(
                f"World-state update for chapter {ch.number} failed ({exc}); keeping the previous state."
            )
            return state.model_copy(
                deep=True,
                update={
                    "chapter_count": ch.number,
                    "last_chapter_hook": fallback_hook or state.last_chapter_hook,
                },
            )

    # ── context ───────────────────────────────────────────────────────────────

    def _base_prefix(self, state: WorldState) -> str:
        b = self.budget
        return (
            f"STYLE GUIDE:\n{fit_text(self.book.style(), b.style)}\n\n"
            f"STORY BIBLE:\n{fit_text(self.book.bible(), b.bible)}\n\n"
            f"WORLD STATE (JSON):\n{fit_text(state.compact(), b.world)}"
        )

    def _recent_chapters(self, before: int) -> list[Chapter]:
        chapters = [c for c in self.book.master().chapters if c.number < before]
        return chapters[-self.cfg.recent_full_count :]

    def _writer_prefix(self, state: WorldState, recent: list[Chapter]) -> str:
        per = self.budget.recent_prose // max(1, len(recent))
        blocks = "\n\n".join(
            f"--- Chapter {c.number}: {c.title} ---\n{fit_text(c.content, per, keep='tail')}"
            for c in recent
        )
        return f"{self._base_prefix(state)}\n\nRECENT CHAPTERS (voice and continuity):\n{blocks or '(none)'}"

    def _recent_summaries(self, before: int) -> str:
        items = [
            f"Ch.{s.number} {s.title}: {s.summary}"
            for s in self._ordered_summaries()
            if s.number < before
        ]
        return "\n".join(fit_items(items[-8:], self.budget.recent_summaries))

    def _rag(self, query: str, exclude: set[int]) -> str:
        hits = self.store.query(query, exclude_chapters=exclude)
        blocks = [f"[Ch.{h.chapter} {h.title}]\n{h.text}" for h in hits]
        return "\n\n".join(fit_items(blocks, self.budget.rag, keep="head"))

    # ── generation ────────────────────────────────────────────────────────────

    def next_chapter_number(self) -> int:
        return self.book.master().last_number + 1

    def generate(self, count: int) -> list[Chapter]:
        self.sync()
        start = self.next_chapter_number()
        pending = self.journal.load()
        if pending is not None and pending.chapter != start:
            self.ui.warn(f"Discarding stale in-progress work for chapter {pending.chapter}.")
            self.journal.finish(pending)
            pending = None
        if pending is not None:
            self.ui.info(f"Resuming chapter {start} from step '{pending.step}'.")

        done: list[Chapter] = []
        costs: list[float] = []
        for i in range(count):
            number = start + i
            if self.meter.max_cost is not None and costs:
                projected = sum(costs) / len(costs)
                if self.meter.total_cost + projected > self.meter.max_cost:
                    self.ui.warn(
                        f"Stopping before chapter {number}: it would likely exceed the "
                        f"${self.meter.max_cost:.2f} budget (about ${projected:.4f} per chapter)."
                    )
                    break
            before = self.meter.total_cost
            done.append(self._generate_one(number, pending if i == 0 else None))
            costs.append(self.meter.total_cost - before)
        if done:
            self._write_outputs(done)
        return done

    def _generate_one(self, number: int, entry: JournalEntry | None) -> Chapter:
        entry = entry or self.journal.start(number)
        spent_before = self.meter.total_cost
        state = self.book.world.load()
        recent = self._recent_chapters(number)
        recent_numbers = {c.number for c in recent}
        base_prefix = self._base_prefix(state)
        writer_prefix = self._writer_prefix(state, recent)
        writer_system = render(
            "writer_system",
            pov=state.narrative_pov or "match the existing chapters",
            tone=state.story_tone or "match the existing chapters",
            humanize=raw("humanize") if self.cfg.humanize else "",
        )
        target = self.cfg.target_words
        target_total = (target[0] + target[1]) // 2
        sep = self.cfg.scene_separator
        recent_text = self._recent_summaries(number)

        if not entry.reached("planned"):
            arcs = self.book.arcs.load()
            query = " ".join(
                [
                    state.last_chapter_hook,
                    state.current_arc,
                    *[t.description for t in state.active_threads[:5]],
                    *[c.name for c in state.characters[:6]],
                ]
            )
            rag = self._rag(query, recent_numbers)
            feedback = ""
            while True:
                self._checkpoint()
                self.ui.step(f"Planning chapter {number}")
                plan_obj = plan_chapter(
                    self.llm,
                    chapter=number,
                    prefix=base_prefix,
                    arc=arcs[-1].summary if arcs else "",
                    recent=recent_text,
                    rag=rag,
                    target_words=target_total,
                    feedback=feedback,
                )
                md = render_plan(plan_obj)
                if self.review is None:
                    break
                action, text = self.review.review_plan(number, md)
                if action == "quit":
                    self.journal.put(number, "plan.md", md)
                    raise GenerationAborted("Stopped during plan review.")
                if action == "regenerate":
                    feedback = text
                    continue
                md = text.strip() or md
                break
            self.journal.put(number, "plan.md", md)
            self.journal.advance(entry, "planned")

        plan = parse_plan(self.journal.get_text(number, "plan.md") or "")
        title = plan.title or f"Chapter {number}"

        while True:
            if not entry.reached("written"):
                scenes: list[str] = self.journal.get_json(number, "scenes.json") or []
                for idx in range(len(scenes), len(plan.scenes)):
                    self._checkpoint()
                    self.ui.step(f"Writing scene {idx + 1}/{len(plan.scenes)}")
                    so_far = join_scenes(scenes, sep)

                    def preview(acc: str, so_far: str = so_far) -> None:
                        self.ui.preview(join_scenes([so_far, acc], sep))

                    text = write_scene(
                        self.llm,
                        chapter=number,
                        plan=plan,
                        index=idx,
                        system=writer_system,
                        prefix=writer_prefix,
                        rag=self._rag(plan.scenes[idx], recent_numbers),
                        so_far=so_far,
                        so_far_tokens=self.budget.so_far,
                        target_total=target_total,
                        last_hook=state.last_chapter_hook,
                        on_preview=preview,
                    )
                    scenes.append(text)
                    self.journal.put(number, "scenes.json", scenes)
                prose = join_scenes(scenes, sep)
                if word_count(prose) < self.cfg.min_words:
                    self._checkpoint()
                    self.ui.step("Extending a short chapter")

                    def preview_extend(acc: str, base: str = prose) -> None:
                        self.ui.preview(f"{base}\n\n{acc}")

                    extra = extend_chapter(
                        self.llm,
                        chapter=number,
                        plan=plan,
                        system=writer_system,
                        prefix=writer_prefix,
                        prose=prose,
                        min_words=self.cfg.min_words,
                        on_preview=preview_extend,
                    )
                    scenes[-1] = f"{scenes[-1]}\n\n{extra}"
                    self.journal.put(number, "scenes.json", scenes)
                self.journal.advance(entry, "written")

            scenes = self.journal.get_json(number, "scenes.json") or []
            if entry.reached("reviewed"):
                break
            if self.cfg.critic:
                scenes = self._critic_pass(
                    number,
                    plan,
                    scenes,
                    writer_system,
                    writer_prefix,
                    base_prefix,
                    recent_text,
                    target_total,
                )
            if self.review is not None:
                action, text = self.review.review_chapter(number, join_scenes(scenes, sep))
                if action == "quit":
                    raise GenerationAborted("Stopped during chapter review.")
                if action == "regenerate":
                    md = self.journal.get_text(number, "plan.md") or plan.markdown
                    if text.strip():
                        md = f"{md.rstrip()}\n\n## Author notes\n{text.strip()}\n"
                        self.journal.put(number, "plan.md", md)
                        plan = parse_plan(md)
                    (self.journal.staging(number) / "scenes.json").unlink(missing_ok=True)
                    self.journal.advance(entry, "planned")
                    continue
                if text.strip() and text.strip() != join_scenes(scenes, sep):
                    scenes = [text.strip()]
                    self.journal.put(number, "scenes.json", scenes)
            self.journal.advance(entry, "reviewed")
            break

        chapter = Chapter(number, title, join_scenes(scenes, sep))
        if not entry.reached("analyzed"):
            self._checkpoint()
            self.ui.step("Updating story memory")
            summary = summarize_chapter(self.llm, chapter)
            new_state = self._next_state(state, chapter, fallback_hook=plan.ending_hook)
            self.journal.put(number, "summary.json", asdict(summary))
            self.journal.put(number, "state.json", new_state.model_dump(mode="json"))
            self.journal.advance(entry, "analyzed")

        self._commit(entry)
        self._maintain(entry)
        self.ui.chapter_done(
            number, title, chapter.word_count, self.meter.total_cost - spent_before
        )
        return self.book.master().by_number().get(number, chapter)

    def _critic_pass(
        self,
        number: int,
        plan_doc: PlanDoc,
        scenes: list[str],
        writer_system: str,
        writer_prefix: str,
        base_prefix: str,
        recent_text: str,
        target_total: int,
    ) -> list[str]:
        sep = self.cfg.scene_separator
        prose = join_scenes(scenes, sep)
        self.ui.step("Continuity check")
        checks = local_checks(
            prose,
            min_words=self.cfg.min_words,
            target=self.cfg.target_words,
            style_guide=self.book.style(),
        )
        if self.cfg.humanize:
            checks += humanize_findings(prose)
        report: CriticReport | None
        try:
            report = critique(
                self.llm,
                chapter=number,
                plan_markdown=plan_doc.markdown,
                prose=prose,
                recent=recent_text,
                checks=checks,
                prefix=base_prefix,
            )
        except (FatalLLMError, BudgetExceeded):
            raise
        except LLMError as exc:
            self.ui.warn(f"Continuity check skipped: {exc}")
            report = None
        if report is not None and report.verdict == "major" and self.cfg.max_rewrites > 0:
            by_scene: dict[int, list[str]] = {}
            for issue in report.issues:
                if (
                    issue.severity == "major"
                    and issue.scene_number
                    and 1 <= issue.scene_number <= len(scenes)
                ):
                    by_scene.setdefault(issue.scene_number - 1, []).append(
                        f"{issue.description} Suggestion: {issue.suggestion}"
                    )
            for idx, issues in sorted(by_scene.items()):
                self._checkpoint()
                self.ui.step(f"Revising scene {idx + 1} ({len(issues)} issue(s))")
                scenes[idx] = rewrite_scene(
                    self.llm,
                    chapter=number,
                    plan=plan_doc,
                    index=idx,
                    scenes=scenes,
                    issues=issues,
                    system=writer_system,
                    prefix=writer_prefix,
                    target_total=target_total,
                )
            self.journal.put(number, "scenes.json", scenes)
        if report is not None and report.verdict != "ok":
            self.ui.warn(f"Continuity check: {report.verdict}, {len(report.issues)} issue(s).")
        self.journal.put(
            number,
            "critic.json",
            {
                "checks": checks,
                "report": report.model_dump(mode="json") if report else None,
            },
        )
        return scenes

    def _commit(self, entry: JournalEntry) -> None:
        number = entry.chapter
        self.journal.advance(entry, "committing")
        plan = parse_plan(self.journal.get_text(number, "plan.md") or "")
        scenes = self.journal.get_json(number, "scenes.json") or []
        title = plan.title or f"Chapter {number}"
        master_book.upsert_chapter(
            self.book.paths.master_book,
            Chapter(number, title, join_scenes(scenes, self.cfg.scene_separator)),
        )
        chapter = self.book.master().by_number()[number]
        self.book.world.save(WorldState.model_validate(self.journal.get_json(number, "state.json")))
        self.store.upsert_chapters([chapter])
        s = self.journal.get_json(number, "summary.json") or {}
        self.book.summaries.put(
            ChapterSummary(number, title, str(s.get("summary", "")), chapter.content_hash)
        )
        meta = self.book.meta()
        if number not in meta.generated:
            meta.generated.append(number)
            self.book.save_meta(meta)

        gen = self.book.paths.generated
        atomic_write_text(
            gen / f"chapter_{number:04d}.md",
            f"# {chapter.display_title}\n\n{chapter.content}\n\n---\n*{chapter.word_count} words*\n",
        )
        atomic_write_text(gen / f"chapter_{number:04d}_plan.md", plan.markdown)
        critic = self.journal.get_json(number, "critic.json")
        if critic is not None:
            atomic_write_text(
                gen / f"chapter_{number:04d}_critic.json",
                json.dumps(critic, indent=2, ensure_ascii=False),
            )
        self.journal.advance(entry, "committed")

    def _maintain(self, entry: JournalEntry) -> None:
        self._sync_arcs()
        if entry.chapter % self.cfg.bible_refresh_every == 0:
            self.ui.step("Refreshing the master bible")
            self.book.save_bible(
                build_bible(self.llm, self.book.world.load(), self.book.arcs.load())
            )
        self.journal.finish(entry)

    # ── outputs ───────────────────────────────────────────────────────────────

    def _write_outputs(self, chapters: list[Chapter]) -> None:
        combined = "\n\n---\n\n".join(f"# {c.display_title}\n\n{c.content}" for c in chapters)
        atomic_write_text(self.book.paths.combined, combined + "\n")
        if self.cfg.export_epub:
            self.export_epub()

    def export_epub(self, output: Path | None = None, template: Path | None = None) -> Path:
        return export_epub(self.book, output, template)


def export_epub(book: Book, output: Path | None = None, template: Path | None = None) -> Path:
    meta = book.meta()
    tpl = template or (book.paths.template_epub if book.paths.template_epub.exists() else None)
    stem = Path(meta.source_name).stem if meta.source_name else book.id
    out = output or book.paths.export_epub(stem)
    write_epub(
        book.master().chapters,
        out,
        tpl,
        EpubMeta(title=meta.title or stem, author=meta.author, language=meta.language),
    )
    return out


def _short(numbers: list[int]) -> str:
    return ", ".join(map(str, numbers[:6])) + ("…" if len(numbers) > 6 else "")
