"""Setting descriptor registry for the ``nextchap config`` browser and setter.

One :class:`SettingSpec` per configurable key, derived automatically from
``Settings.model_fields``: a new pydantic field in ``config.py`` shows up here
without any edits. A field type that has no :data:`Kind` mapping fails loudly
at import instead of silently disappearing from the UI.

Defaults are pristine declared defaults (``Settings.model_construct()``), not
the process's effective values — env vars, ``.env`` and the user's config.toml
must not leak into the registry. Consumers wanting the current effective value
create their own ``Settings()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import NoneType, UnionType
from typing import Any, Literal, Union, get_args, get_origin

from annotated_types import Ge, Le
from pydantic import BaseModel, SecretStr
from pydantic.fields import FieldInfo

from nextchap.config import Settings

__all__ = [
    "DESCRIPTORS",
    "DESCRIPTORS_BY_KEY",
    "SECTIONS",
    "SECTION_TITLES",
    "Kind",
    "SettingSpec",
    "is_setting_key",
    "section_keys",
    "spec",
]

Kind = Literal[
    "str", "int", "float", "bool", "enum", "list[float]", "list[str]", "model", "model_list"
]

SECTIONS: tuple[str, ...] = (
    "models",
    "defaults",
    "generation",
    "retrieval",
    "ui",
    "cleanup",
    "network",
    "security",
)

SECTION_TITLES: dict[str, str] = {
    "models": "Models per role",
    "defaults": "Command defaults",
    "generation": "Generation behaviour",
    "retrieval": "Retrieval and context",
    "ui": "Terminal UI",
    "cleanup": "Text cleanup",
    "network": "Network and API",
    "security": "API key",
    "data_dir": "Data location",
}


@dataclass(frozen=True, slots=True)
class SettingSpec:
    """One user-editable setting: identity, kind, live default, and help text."""

    key: str  # dotted, e.g. "models.writer.temperature"
    section: str  # top-level Settings field name, e.g. "models"
    kind: Kind
    default: Any  # live value from Settings()
    help: str  # one short plain-English line, user-facing
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    secret: bool = False
    editable: bool = True
    optional: bool = False  # field is X | None; blank input means "unset"


# Help text for the per-role model fields; the role name is interpolated.
_ROLE_HELP: dict[str, str] = {
    "model": "OpenRouter model id used for the {role} role.",
    "reasoning": "Reasoning effort for the {role} model (none = off).",
    "temperature": "Sampling temperature for the {role} model (unset = provider default).",
    "top_p": "Nucleus-sampling top_p for the {role} model (unset = provider default).",
    "max_tokens": "Max output tokens per request for the {role} model.",
    "fallbacks": "Model ids tried in order when the {role} model fails.",
}

# Help text for every non-model setting, keyed by its full dotted key.
_HELP: dict[str, str] = {
    "api_key": "OpenRouter API key — read from env or keyring, never written to disk.",
    "data_dir": "Where books, logs and other data live (default: current directory).",
    "defaults.chapters": "How many chapters one `generate` run writes by default.",
    "defaults.max_cost": "Default spending cap in USD for a generate run (blank = no cap).",
    "defaults.review": "Pause for review after each written chapter by default.",
    "defaults.preview": "Print a chapter preview before committing by default.",
    "defaults.critic": "Run the critic pass by default (blank = follow [generation].critic).",
    "defaults.export_epub": "Export an EPUB after generating (blank = follow [generation]).",
    "generation.target_words": "Two positive integers: min and max words per chapter (e.g. 1500, 2000).",
    "generation.min_words": "Warn when a written chapter falls below this word count.",
    "generation.arc_size": "How many chapters form one story arc.",
    "generation.recent_full_count": "How many recent chapters are included in full as context.",
    "generation.bible_refresh_every": "Refresh the story bible every N chapters.",
    "generation.max_prompt_tokens": "Cap on prompt size in tokens; older context is trimmed first.",
    "generation.critic": "Run the critic pass after writing each chapter.",
    "generation.humanize": "Run a naturalness pass over drafted prose.",
    "generation.max_rewrites": "How many times a chapter may be rewritten after review.",
    "generation.max_continuations": "How many continuation attempts when a chapter comes up short.",
    "generation.max_cost": "Hard spending cap in USD for a generate run (blank = no cap).",
    "generation.scene_separator": "Marker inserted between scenes in the drafted text.",
    "generation.export_epub": "Write an EPUB export after a successful chapter commit.",
    "retrieval.embed_backend": "Embedding backend: local, remote, or hash (testing only).",
    "retrieval.embed_model": "Local embedding model id (changing it needs fix --rebuild-index).",
    "retrieval.remote_embed_model": "OpenRouter model id used when embed_backend is remote.",
    "retrieval.chunk_words": "Target words per indexed chunk.",
    "retrieval.chunk_overlap_words": "Words shared between neighbouring chunks.",
    "retrieval.top_k": "How many chunks are retrieved into the writing context.",
    "retrieval.candidates": "Candidate pool size before reranking and filtering.",
    "retrieval.hybrid": "Combine dense and keyword search (BM25) with RRF.",
    "retrieval.dense_weight": "Weight of dense search vs keyword search in the blend.",
    "retrieval.rerank": "Rerank retrieved chunks with a cross-encoder model.",
    "retrieval.rerank_model": "Local reranker model id used when rerank is on.",
    "ui.footer_icon": "Show the Nerd Font icon in the status footer.",
    "cleanup.extra_noise_patterns": "Extra regex patterns stripped from imported chapter text.",
    "network.base_url": "OpenRouter API base URL.",
    "network.connect_timeout": "Seconds to wait while connecting to the API.",
    "network.read_timeout": "Seconds to wait for an API response.",
    "network.max_attempts": "Retry attempts for transient API failures.",
    "network.app_title": "App title sent to OpenRouter for ranking.",
    "network.app_url": "App URL sent to OpenRouter for ranking.",
}


def _unwrap_none(annotation: Any) -> tuple[Any, bool]:
    """Return ``(inner type, optional flag)`` for ``X | None`` annotations."""
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        args = get_args(annotation)
        non_none = [a for a in args if a is not NoneType]
        if NoneType in args and len(non_none) == 1:
            return non_none[0], True
    return annotation, False


def _kind_of(t: Any) -> Kind:
    """Map a pydantic field type to a :data:`Kind`."""
    if get_origin(t) is Literal:
        return "enum"
    if t is bool:
        return "bool"
    if t is int:
        return "int"
    if t is float:
        return "float"
    if t is SecretStr or t is str or t is Path:
        return "str"
    origin = get_origin(t)
    args = get_args(t)
    if origin is list and args == (float,):
        return "list[float]"
    if origin is list and args == (str,):
        return "list[str]"
    if origin is tuple and args == (int, int):
        # tuple[int, int] (generation.target_words) is edited as a comma-separated
        # pair of integers and round-trips through Settings validation.
        return "list[float]"
    raise TypeError(f"no Kind mapping for field type {t!r}; add one in config_schema.py")


def _limits(field: FieldInfo) -> tuple[float | None, float | None]:
    """Read ge/le constraints off the pydantic field metadata."""
    minimum: float | None = None
    maximum: float | None = None
    for meta in field.metadata:
        bound: Any
        if isinstance(meta, Ge):
            bound = meta.ge
            minimum = float(bound)
        elif isinstance(meta, Le):
            bound = meta.le
            maximum = float(bound)
    return minimum, maximum


def _help_for(key: str) -> str:
    if key.startswith("models."):
        role, leaf = key.split(".")[1], key.rsplit(".", 1)[1]
        template = _ROLE_HELP.get(leaf)
        if template is not None:
            return template.format(role=role)
    return _HELP.get(key) or f"Configure {key.replace('_', ' ')}."


def _leaf_spec(key: str, section: str, field: FieldInfo, default: Any) -> SettingSpec:
    inner, optional = _unwrap_none(field.annotation)
    kind = _kind_of(inner)
    if kind == "str" and (key.endswith(".model") or key.endswith("_model")):
        kind = "model"
    elif kind == "list[str]" and key.endswith(".fallbacks"):
        kind = "model_list"
    minimum, maximum = _limits(field)
    return SettingSpec(
        key=key,
        section=section,
        kind=kind,
        default=default,
        help=_help_for(key),
        choices=tuple(get_args(inner)) if kind == "enum" else (),
        minimum=minimum,
        maximum=maximum,
        secret=key == "api_key",
        editable=key != "api_key",
        optional=optional,
    )


def _walk(prefix: str, section: str, field: FieldInfo, live: Any) -> list[SettingSpec]:
    """Recurse into nested BaseModel fields and collect one spec per leaf."""
    inner, _ = _unwrap_none(field.annotation)
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        specs: list[SettingSpec] = []
        for leaf, leaf_field in inner.model_fields.items():
            child_live = None if live is None else getattr(live, leaf)
            specs.extend(
                _walk(f"{prefix}.{leaf}" if prefix else leaf, section, leaf_field, child_live)
            )
        return specs
    if prefix == "api_key":
        # Never let the live secret (env or keyring) leak into the registry.
        return [_leaf_spec(prefix, "security", field, None)]
    return [_leaf_spec(prefix, section, field, live)]


def _build() -> tuple[SettingSpec, ...]:
    """Walk ``Settings.model_fields`` and collect one spec per leaf key."""
    # model_construct, not Settings(): skip env/dotenv/toml sources so the
    # developer's personal config never leaks into the registry defaults.
    specs: list[SettingSpec] = []
    for name, field in Settings.model_fields.items():
        specs.extend(_walk(name, name, field, getattr(_LIVE, name)))
    # Stable sort: sections in SECTIONS order, intra-section declaration order
    # kept; extra sections (data_dir) land at the end.
    order = {name: i for i, name in enumerate(SECTIONS)}
    specs.sort(key=lambda s: order.get(s.section, len(SECTIONS)))
    return tuple(specs)


_LIVE: Settings = Settings.model_construct()

DESCRIPTORS: tuple[SettingSpec, ...] = _build()
DESCRIPTORS_BY_KEY: dict[str, SettingSpec] = {s.key: s for s in DESCRIPTORS}


def spec(key: str) -> SettingSpec:
    """Return the descriptor for ``key``; raises KeyError when unknown."""
    try:
        return DESCRIPTORS_BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown setting key: {key!r}") from None


def section_keys(section: str) -> tuple[str, ...]:
    """All setting keys in ``section``, in declaration order."""
    return tuple(s.key for s in DESCRIPTORS if s.section == section)


def is_setting_key(key: str) -> bool:
    """True when ``key`` is a known setting key."""
    return key in DESCRIPTORS_BY_KEY
