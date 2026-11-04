"""Versioned prompt templates (`$name` placeholders, string.Template syntax)."""

from __future__ import annotations

from functools import cache
from importlib import resources
from string import Template

PROMPT_VERSION = "2.1.0"


@cache
def _template(name: str) -> Template:
    text = resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
    return Template(text)


def render(name: str, **values: object) -> str:
    return _template(name).substitute({k: str(v) for k, v in values.items()}).strip()


def raw(name: str) -> str:
    return resources.files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
