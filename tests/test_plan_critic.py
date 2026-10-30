from __future__ import annotations

from nextchap.pipeline.critic import banned_phrases, local_checks
from nextchap.pipeline.plan import ChapterPlan, Scene, parse_plan, render_plan
from nextchap.pipeline.writer import clean_prose, join_scenes
from nextchap.prompts import PROMPT_VERSION, raw, render


def make_plan() -> ChapterPlan:
    return ChapterPlan(
        title="The Ember Gate",
        opening_hook="Smoke on the stairs.",
        scenes=[
            Scene(
                location="Gate",
                characters=["Lin"],
                purpose="p",
                beats=["b1", "b2"],
                target_words=600,
            ),
            Scene(location="Hall", characters=["Mei"], purpose="q", beats=["c1"], target_words=900),
        ],
        threads_advanced=["t001"],
        character_moment="m",
        revelations=[],
        ending_hook="The gate opens.",
        tone="tense",
        continuity_traps=["Lin's left arm is broken"],
        must_dramatize=["duel"],
    )


def test_plan_roundtrip() -> None:
    doc = parse_plan(render_plan(make_plan()))
    assert doc.title == "The Ember Gate"
    assert len(doc.scenes) == 2 and doc.scenes[1].startswith("Scene 2: Hall")
    assert doc.scene_words == [600, 900]
    assert doc.ending_hook == "The gate opens."
    assert doc.opening_hook == "Smoke on the stairs."


def test_hand_edited_plan_without_scenes() -> None:
    doc = parse_plan("# My title\n\nJust write a duel on the bridge.")
    assert doc.title == "My title"
    assert len(doc.scenes) == 1 and doc.scene_words == [None]


def test_clean_prose_strips_labels() -> None:
    raw_text = "Here is the scene you asked for:\n\n**Scene 2: The Hall**\n\nMei laughed.\n\n## Note\nShe left.\n\n---"
    assert clean_prose(raw_text) == "Mei laughed.\n\nShe left."


def test_join_scenes() -> None:
    assert join_scenes(["a", " ", "b"], "* * *") == "a\n\n* * *\n\nb"
    assert join_scenes(["a", "b"], "") == "a\n\nb"


def test_local_checks() -> None:
    style = 'Avoid these overused phrases: "a testament to", "little did they know".\nOther "quoted" line.'
    assert banned_phrases(style) == ["a testament to", "little did they know"]
    prose = "It was a testament to his will. " + "the cold wind cut through the valley again " * 4
    found = local_checks(prose, min_words=900, target=(1500, 2000), style_guide=style)
    assert any("below the minimum" in f for f in found)
    assert any("Repeated phrasing" in f for f in found)
    assert any("a testament to" in f for f in found)


def test_prompts_render() -> None:
    assert PROMPT_VERSION
    assert "Chapter 7" in render("summarize_chapter", number=7, title="T", content="x")
    assert "$" not in render(
        "planner", chapter=1, target_words=1750, arc="a", recent="r", rag="g", feedback=""
    )
    assert raw("style_default").startswith("# Style guide")
