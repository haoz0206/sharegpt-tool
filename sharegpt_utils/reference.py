"""Indexed media references: ``<@image:N>``, ``<@video:N>``, ``<@audio:N>``.

A reference is a dedicated sentinel token with explicit pool + index. It
resolves against the already-built top-level media pool (same pool used by
the legacy ``<image>`` / ``<video>`` / ``<audio>`` tokens), giving random
access where the legacy tokens only supported ordered consumption.

Two entry points:

* :func:`substitute_references_in_text` — substring substitution over one
  string. Used by the prompt parser before it splits on legacy tokens.
* :func:`resolve_references_in_obj` — recursive walker over a dict/list/str
  tree. Used by the parser's ``extra_info`` pass in agent-style datasets.

Both honor the ``<@kind:N>`` anchor and leave any other text untouched, so
the default (legacy) parse path is fully preserved.
"""

from __future__ import annotations

import re
from typing import Any, Callable

OnReference = Callable[[str, int], None]

REFERENCE_PATTERN = re.compile(r"<@(image|video|audio):(\d+)>")


class ReferenceResolutionError(ValueError):
    """Raised when a reference cannot be resolved (e.g., index out of range)."""


def _lookup(
    pool: dict[str, list[Any]],
    media_type: str,
    index: int,
    *,
    json_path: str,
    strict: bool,
) -> Any | None:
    """Look up ``pool[media_type][index]`` or fail/skip per strict flag."""
    items = pool.get(media_type, [])
    if 0 <= index < len(items):
        return items[index]
    if strict:
        raise ReferenceResolutionError(
            f"Unresolvable reference <@{media_type}:{index}> at {json_path}: "
            f"{media_type} pool has {len(items)} item(s)."
        )
    return None


def substitute_references_in_text(
    text: str,
    pool: dict[str, list[Any]],
    *,
    strict: bool = True,
    json_path: str = "<text>",
    on_reference: OnReference | None = None,
) -> str:
    """Replace every ``<@kind:N>`` substring in ``text`` with its pool value.

    Only rewrites exact ``<@kind:N>`` matches; surrounding text is kept as-is.
    Non-string pool entries (e.g., PIL objects) are stringified by ``re.sub``.
    """

    def _replace(match: re.Match[str]) -> str:
        media_type = match.group(1)
        index = int(match.group(2))
        resolved = _lookup(
            pool,
            media_type,
            index,
            json_path=json_path,
            strict=strict,
        )
        if resolved is None:
            return match.group(0)
        if on_reference is not None:
            on_reference(media_type, index)
        return str(resolved)

    return REFERENCE_PATTERN.sub(_replace, text)


def _is_exact_reference(value: str) -> bool:
    return REFERENCE_PATTERN.fullmatch(value) is not None


def resolve_references_in_obj(
    obj: Any,
    pool: dict[str, list[Any]],
    *,
    strict: bool = True,
    path: str = "$",
    on_reference: OnReference | None = None,
) -> Any:
    """Walk ``obj`` (dict/list/str) and resolve ``<@kind:N>`` references.

    Strings that are exactly a reference (``"<@image:0>"``) are replaced by
    the resolved pool value (which may be any type — typically an absolute
    path string, but non-string objects pass through unchanged). Strings
    that contain a reference as a substring have only the matching segments
    rewritten via :func:`substitute_references_in_text`.

    Returns the transformed object. Dicts and lists are rebuilt rather than
    mutated so callers can swap in the result without aliasing concerns.
    Non-container scalars that are not strings are returned as-is.
    """
    if isinstance(obj, str):
        if _is_exact_reference(obj):
            match = REFERENCE_PATTERN.fullmatch(obj)
            assert match is not None  # _is_exact_reference guard
            media_type = match.group(1)
            index = int(match.group(2))
            resolved = _lookup(pool, media_type, index, json_path=path, strict=strict)
            if resolved is None:
                return obj
            if on_reference is not None:
                on_reference(media_type, index)
            return resolved
        return substitute_references_in_text(
            obj, pool, strict=strict, json_path=path, on_reference=on_reference
        )
    if isinstance(obj, dict):
        return {
            key: resolve_references_in_obj(
                value,
                pool,
                strict=strict,
                path=f"{path}.{key}",
                on_reference=on_reference,
            )
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [
            resolve_references_in_obj(
                item,
                pool,
                strict=strict,
                path=f"{path}[{i}]",
                on_reference=on_reference,
            )
            for i, item in enumerate(obj)
        ]
    return obj


__all__ = [
    "REFERENCE_PATTERN",
    "ReferenceResolutionError",
    "substitute_references_in_text",
    "resolve_references_in_obj",
]
