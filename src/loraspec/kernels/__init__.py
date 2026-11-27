"""Per-token Top-k Arrow routing, @triton.jit Arrow-SGMV kernels, reference & roofline bench."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Any


def optional_torch() -> Any | None:
    """Dynamically import `torch` if installed, else return None."""
    if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
        return None
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def optional_triton() -> tuple[Any, Any] | tuple[None, None]:
    """Dynamically import `(triton, triton.language)` if installed, else `(None, None)`."""
    if "triton" not in sys.modules and importlib.util.find_spec("triton") is None:
        return None, None
    try:
        triton_mod = importlib.import_module("triton")
        tl_mod = importlib.import_module("triton.language")
    except ImportError:
        return None, None
    return triton_mod, tl_mod


def has_torch() -> bool:
    """Return True if PyTorch is importable in the current environment."""
    return optional_torch() is not None


def has_mps() -> bool:
    """Return True if Apple Metal Performance Shaders (MPS) is available via PyTorch."""
    torch_mod = optional_torch()
    if torch_mod is None:
        return False
    try:
        backends: Any = getattr(torch_mod, "backends", None)
        mps: Any = getattr(backends, "mps", None)
        return bool(mps is not None and mps.is_available())
    except Exception:
        return False


def has_triton_cuda() -> bool:
    """Return True if both CUDA via PyTorch and Triton are available."""
    torch_mod = optional_torch()
    if torch_mod is None:
        return False
    try:
        cuda: Any = getattr(torch_mod, "cuda", None)
        if cuda is None or not bool(cuda.is_available()):
            return False
    except Exception:
        return False
    triton_mod, tl_mod = optional_triton()
    return triton_mod is not None and tl_mod is not None


from loraspec.kernels.arrow_sgmv import (  # noqa: E402
    SegmentLayout,
    build_segment_metadata,
    triton_arrow_sgmv_forward,
)
from loraspec.kernels.bench import run_kernel_parity_and_roofline_bench  # noqa: E402
from loraspec.kernels.dispatch import ArrowMoLoRADispatcher  # noqa: E402
from loraspec.kernels.reference import (  # noqa: E402
    arrow_molora_reference,
    peft_eager_arrow_forward,
    pure_python_arrow_reference,
    static_lora_forward,
)
from loraspec.kernels.router import (  # noqa: E402
    ArrowPrototypeBank,
    compute_router_entropy,
    extract_arrow_prototypes,
    route_topk_tokens,
)

__all__ = [
    "ArrowMoLoRADispatcher",
    "ArrowPrototypeBank",
    "SegmentLayout",
    "arrow_molora_reference",
    "build_segment_metadata",
    "compute_router_entropy",
    "extract_arrow_prototypes",
    "has_mps",
    "has_torch",
    "has_triton_cuda",
    "optional_torch",
    "optional_triton",
    "peft_eager_arrow_forward",
    "pure_python_arrow_reference",
    "route_topk_tokens",
    "run_kernel_parity_and_roofline_bench",
    "static_lora_forward",
    "triton_arrow_sgmv_forward",
]
