"""The tool registry: every callable an agent can be granted.

Agents name the tools they want as strings; `resolve` turns those into the
objects passed to the model. An unknown name raises rather than silently
producing an under-equipped agent.
"""

from __future__ import annotations

from typing import Any

from . import delegate, gmail, memory, notes, search, youtube

_REGISTRY: dict[str, Any] = {
    **gmail.TOOLS,
    **notes.TOOLS,
    **youtube.TOOLS,
    **search.TOOLS,
    **memory.TOOLS,
    **delegate.TOOLS,
}


def available() -> list[str]:
    """Every registered tool name, sorted."""
    return sorted(_REGISTRY)


def resolve(names: list[str]) -> list[Any]:
    """Look up tool objects by name, preserving the order given."""
    unknown = [name for name in names if name not in _REGISTRY]
    if unknown:
        raise KeyError(
            f"Unknown tool(s): {', '.join(sorted(unknown))}. "
            f"Registered tools: {', '.join(available())}"
        )
    return [_REGISTRY[name] for name in names]


__all__ = ["available", "resolve"]
