"""Torch device selection, shared by the embedder and the reranker."""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str | None = None) -> str:
    """Pick a torch device.

    Args:
        requested: An explicit device string. Returned unchanged when given,
            so callers can force ``"cpu"`` in tests or on constrained hosts.

    Returns:
        ``"cuda"``, ``"mps"``, or ``"cpu"``. Falls back to ``"cpu"`` if torch
        is unavailable or its backend probes raise.
    """
    if requested:
        return requested

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dependency
        logger.warning("torch not importable; falling back to CPU")
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001 - probing must never break startup
        logger.debug("CUDA probe failed", exc_info=True)

    # torch.backends.mps is absent on older builds and can raise on non-Apple
    # hardware, so guard both the attribute and the call.
    try:
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        logger.debug("MPS probe failed", exc_info=True)

    return "cpu"
