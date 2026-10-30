"""Strict JSON Schema generation and lenient JSON extraction."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from pydantic import BaseModel

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _strictify(node: Any) -> Any:
    if isinstance(node, dict):
        node.pop("default", None)
        node.pop("title", None)
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for value in node:
            _strictify(value)
    return node


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Schema in the subset accepted by strict structured-output providers."""
    schema: dict[str, Any] = _strictify(copy.deepcopy(model.model_json_schema()))
    return schema


def response_format(model: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": model.__name__,
            "strict": True,
            "schema": strict_json_schema(model),
        },
    }


def extract_json(text: str) -> str:
    """Return the JSON object in a reply, tolerating code fences and stray prose."""
    s = _FENCE_RE.sub("", text.strip()).strip()
    if s.startswith("{"):
        return s
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        return s[start : end + 1]
    return s


def schema_prompt(model: type[BaseModel]) -> str:
    return (
        "Respond with ONLY a JSON object (no prose, no code fences) that matches this JSON Schema:\n"
        + json.dumps(strict_json_schema(model), ensure_ascii=False)
    )
