from __future__ import annotations

from typing import List, Optional


def get_best_device() -> str:
    """Choose best torch device with priority: mps -> cuda -> cpu."""
    try:
        import torch
    except Exception:
        return "cpu"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_device(device: Optional[str] = None) -> str:
    if device is None:
        return get_best_device()

    norm = str(device).strip().lower()
    if norm in {"mps", "cuda", "cpu"}:
        return norm
    raise ValueError(f"Unsupported device '{device}'. Use one of: mps, cuda, cpu.")


def insightface_providers(device: Optional[str] = None) -> List[str]:
    """
    ONNX Runtime provider order for InsightFace.
    MPS is not an ONNX Runtime provider, so mps falls back to CPU for InsightFace.
    """
    resolved = resolve_device(device)
    if resolved == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def insightface_ctx_id(device: Optional[str] = None) -> int:
    """InsightFace ctx_id: 0 for CUDA, -1 for CPU execution."""
    resolved = resolve_device(device)
    return 0 if resolved == "cuda" else -1
