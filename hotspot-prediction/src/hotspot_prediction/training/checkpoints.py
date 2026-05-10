"""Checkpoint I/O helpers."""

from __future__ import annotations

from typing import Any, Callable


def load_checkpoint(
    path: str,
    map_location: Any = None,
    torch_load: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Load a PyTorch checkpoint using torch.load or an injected loader."""
    if torch_load is None:
        import torch

        torch_load = torch.load

    checkpoint = torch_load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict from {path!r}, got {type(checkpoint).__name__}")
    return checkpoint
