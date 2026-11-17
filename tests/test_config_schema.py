"""Tests for the setting descriptor registry (no network, no model calls)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from nextchap.config import RoleModel, Settings
from nextchap.config_schema import (
    DESCRIPTORS,
    DESCRIPTORS_BY_KEY,
    SECTION_TITLES,
    SECTIONS,
    SettingSpec,
    is_setting_key,
    section_keys,
    spec,
)


def _inner(annotation: Any) -> Any:
    """Strip one ``X | None`` layer (test-side mirror of the registry logic)."""
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not NoneType]
        if NoneType in args and len(non_none) == 1:
            return non_none[0]
    return annotation


def _leaves() -> dict[str, Any]:
    """Flatten ``Settings().model_dump()`` into {dotted key: leaf value}."""

    def walk(prefix: str, obj: dict[str, Any]) -> Any:
        for key, value in obj.items():
            dotted = f"{prefix}{key}"
            if isinstance(value, dict):
                yield from walk(f"{dotted}.", value)
            else:
                yield dotted, value

    return dict(walk("", Settings().model_dump()))


def _literal_args() -> dict[str, tuple[Any, ...]]:
    """Every Literal annotation in Settings, keyed by dotted key."""
    out: dict[str, tuple[Any, ...]] = {}

    def visit(prefix: str, annotation: Any) -> None:
        ann = _inner(annotation)
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            for leaf, leaf_field in ann.model_fields.items():
                visit(f"{prefix}.{leaf}" if prefix else leaf, leaf_field.annotation)
        elif get_origin(ann) is Literal:
            out[prefix] = tuple(get_args(ann))

    for name, field in Settings.model_fields.items():
        visit(name, field.annotation)
    return out


def _ge_constraints() -> dict[str, float]:
    """Every ``ge`` constraint in Settings, keyed by dotted key."""
    out: dict[str, float] = {}
    for name, field in Settings.model_fields.items():
        ann = _inner(field.annotation)
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            for leaf, leaf_field in ann.model_fields.items():
                for meta in leaf_field.metadata:
                    if hasattr(meta, "ge"):
                        out[f"{name}.{leaf}"] = float(meta.ge)
        else:
            for meta in field.metadata:
                if hasattr(meta, "ge"):
                    out[name] = float(meta.ge)
    return out


def test_key_coverage_both_directions() -> None:
    leaves = _leaves()
    spec_keys = {s.key for s in DESCRIPTORS}
    assert spec_keys == set(leaves), (
        f"missing specs: {sorted(set(leaves) - spec_keys)}, "
        f"stale specs: {sorted(spec_keys - set(leaves))}"
    )


def test_defaults_match_live_dump() -> None:
    leaves = _leaves()
    for s in DESCRIPTORS:
        if s.key == "api_key":
            # The registry must never carry the live secret, even if one is set.
            assert s.default is None
        else:
            assert s.default == leaves[s.key], s.key


def test_every_spec_has_help() -> None:
    for s in DESCRIPTORS:
        assert s.help and len(s.help) <= 90, s.key


def test_enum_specs_match_literal_args() -> None:
    literals = _literal_args()
    for s in DESCRIPTORS:
        if s.kind == "enum":
            assert s.choices, s.key
            assert s.key in literals, s.key
            assert s.choices == tuple(literals[s.key]), s.key
        else:
            assert s.choices == (), s.key


def test_ge_constraints_read_from_metadata() -> None:
    ges = _ge_constraints()
    for s in DESCRIPTORS:
        if s.key in ges:
            assert s.minimum == ges[s.key], s.key
        else:
            assert s.minimum is None, s.key
    assert "defaults.chapters" in ges
    assert "defaults.max_cost" in ges
    for s in DESCRIPTORS:
        assert s.maximum is None, s.key  # no le constraints exist today


def test_descriptors_order_matches_sections() -> None:
    sections = [s.section for s in DESCRIPTORS]
    seen = list(dict.fromkeys(sections))
    assert [x for x in seen if x in SECTIONS] == list(SECTIONS)
    # data_dir is a top-level leaf whose section is its own field name; it is
    # not browsable via SECTIONS and lands at the end of DESCRIPTORS.
    assert set(seen) - set(SECTIONS) == {"data_dir"}
    assert seen[-1] == "data_dir"
    assert set(SECTION_TITLES) == set(SECTIONS) | {"data_dir"}
    assert all(SECTION_TITLES.values())
    assert len(DESCRIPTORS_BY_KEY) == len(DESCRIPTORS)
    assert len({s.key for s in DESCRIPTORS}) == len(DESCRIPTORS)


def test_kind_spot_checks() -> None:
    assert spec("models.writer.model").kind == "model"
    assert spec("models.writer.fallbacks").kind == "model_list"
    assert spec("models.planner.reasoning").kind == "enum"
    assert spec("models.planner.reasoning").choices == ("none", "minimal", "low", "medium", "high")
    assert spec("models.planner.reasoning").optional is True
    assert spec("models.writer.temperature").kind == "float"
    assert spec("models.writer.temperature").optional is True
    assert spec("models.analyst.max_tokens").kind == "int"
    assert spec("models.analyst.max_tokens").optional is False
    assert spec("retrieval.embed_model").kind == "model"
    assert spec("retrieval.remote_embed_model").kind == "model"
    assert spec("retrieval.rerank_model").kind == "model"
    assert spec("retrieval.embed_backend").kind == "enum"
    assert spec("retrieval.embed_backend").choices == ("local", "remote", "hash")
    assert spec("defaults.chapters").kind == "int"
    assert spec("defaults.chapters").minimum == 1
    assert spec("defaults.max_cost").kind == "float"
    assert spec("defaults.max_cost").minimum == 0
    assert spec("defaults.max_cost").optional is True
    assert spec("defaults.critic").kind == "bool"
    assert spec("defaults.critic").optional is True
    assert spec("defaults.export_epub").kind == "bool"
    assert spec("defaults.export_epub").optional is True
    assert spec("defaults.review").kind == "bool"
    assert spec("defaults.review").optional is False
    assert spec("generation.target_words").kind == "list[float]"
    assert spec("generation.critic").kind == "bool"
    assert spec("generation.critic").optional is False
    assert spec("generation.max_cost").kind == "float"
    assert spec("generation.max_cost").optional is True
    assert spec("generation.max_cost").minimum is None
    assert spec("generation.min_words").kind == "int"
    assert spec("ui.footer_icon").kind == "bool"
    assert spec("cleanup.extra_noise_patterns").kind == "list[str]"
    assert spec("network.connect_timeout").kind == "float"
    assert spec("network.base_url").kind == "str"
    assert spec("data_dir").kind == "str"
    assert spec("data_dir").optional is True
    assert spec("data_dir").section == "data_dir"


def test_api_key_spec() -> None:
    s = spec("api_key")
    assert s.section == "security"
    assert s.kind == "str"
    assert s.secret is True
    assert s.editable is False
    assert s.optional is True
    assert s.default is None
    assert "never written to disk" in s.help


def test_target_words_round_trips_through_settings() -> None:
    # list[float] of two integral floats must survive Settings validation.
    settings = Settings(generation={"target_words": [1500.0, 2000.0]})
    assert settings.generation.target_words == (1500, 2000)
    with pytest.raises(ValidationError):
        Settings(generation={"target_words": [1500.0, 2000.0, 2500.0]})


def test_spec_lookup_and_helpers() -> None:
    with pytest.raises(KeyError):
        spec("nope")
    assert is_setting_key("models.writer.model")
    assert is_setting_key("defaults.chapters")
    assert not is_setting_key("nope")
    assert not is_setting_key("models.writer")  # group key, not a leaf
    role_fields = tuple(RoleModel.model_fields)
    expected = tuple(
        f"models.{role}.{leaf}"
        for role in ("writer", "planner", "analyst", "critic")
        for leaf in role_fields
    )
    assert section_keys("models") == expected
    assert section_keys("ui") == ("ui.footer_icon",)
    assert section_keys("nope") == ()


def test_specs_are_frozen() -> None:
    s = spec("ui.footer_icon")
    with pytest.raises(FrozenInstanceError):
        s.default = False  # type: ignore[misc]


def test_spec_type_is_dataclass_with_slots() -> None:
    assert isinstance(spec("ui.footer_icon"), SettingSpec)
    assert not hasattr(spec("ui.footer_icon"), "__dict__")
