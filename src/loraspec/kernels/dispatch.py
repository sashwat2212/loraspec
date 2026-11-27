"""Cross-platform Arrow-MoLoRA auto-dispatcher across CUDA Triton, MPS, CPU & pure Python."""

from __future__ import annotations

from typing import Any, Literal, cast

from loraspec.core.errors import ConfigError, RuntimeExecutionError
from loraspec.core.schemas import MoLoRARouteTelemetryBatch
from loraspec.kernels import has_mps, has_triton_cuda, optional_torch
from loraspec.kernels.arrow_sgmv import build_segment_metadata, triton_arrow_sgmv_forward
from loraspec.kernels.reference import arrow_molora_reference, pure_python_arrow_reference
from loraspec.kernels.router import route_topk_tokens

DispatchBackend = Literal[
    "auto",
    "triton_cuda",
    "torch_mps",
    "torch_cpu",
    "torch_reference",
    "pure_python",
]

VALID_DISPATCH_BACKENDS: frozenset[str] = frozenset(
    {
        "auto",
        "triton_cuda",
        "torch_mps",
        "torch_cpu",
        "torch_reference",
        "pure_python",
    }
)


def _is_torch_tensor(value: Any) -> bool:
    torch_mod = optional_torch()
    return torch_mod is not None and isinstance(value, torch_mod.Tensor)


class ArrowMoLoRADispatcher:
    """Unified runtime dispatcher selecting Triton CUDA, MPS, PyTorch CPU, or pure Python."""

    def forward(
        self,
        x: Any,
        base_out: Any,
        prototypes: Any,
        a_weights: Any,
        b_weights: Any,
        ranks: Any,
        scalings: Any,
        top_k: int = 2,
        temperature: float = 1.0,
        backend: DispatchBackend = "auto",
        *,
        layer_name: str | None = None,
    ) -> tuple[Any, MoLoRARouteTelemetryBatch]:
        """Execute per-token Top-k Arrow-MoLoRA forward pass on the resolved backend."""
        if backend not in VALID_DISPATCH_BACKENDS:
            raise ConfigError(
                f"Unsupported dispatch backend '{backend}'. "
                f"Expected one of {sorted(VALID_DISPATCH_BACKENDS)}."
            )
        resolved_layer = layer_name or "molora_layer"

        if backend == "pure_python":
            return self._run_pure_python(
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        if backend == "triton_cuda":
            if not has_triton_cuda():
                raise RuntimeExecutionError(
                    "Requested backend='triton_cuda', but CUDA + Triton are unavailable on "
                    "this host. Use backend='auto' or install `loraspec[kernels]` on Linux CUDA."
                )
            return self._run_triton_cuda(
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        if backend == "torch_mps":
            if not has_mps():
                raise RuntimeExecutionError(
                    "Requested backend='torch_mps', but Apple MPS is unavailable on this host."
                )
            torch_mod = optional_torch()
            assert torch_mod is not None
            return self._run_torch_on_device(
                torch_mod=torch_mod,
                target_device="mps",
                backend_label="torch_mps",
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        if backend == "torch_cpu":
            torch_mod = optional_torch()
            if torch_mod is None:
                raise RuntimeExecutionError(
                    "Requested backend='torch_cpu', but PyTorch is not installed."
                )
            return self._run_torch_on_device(
                torch_mod=torch_mod,
                target_device="cpu",
                backend_label="torch_cpu",
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        if backend == "torch_reference":
            torch_mod = optional_torch()
            if torch_mod is None:
                raise RuntimeExecutionError(
                    "Requested backend='torch_reference', but PyTorch is not installed."
                )
            if _is_torch_tensor(x):
                dev_label: Literal["torch_mps", "torch_cpu"] = (
                    "torch_mps" if str(x.device.type).lower() == "mps" else "torch_cpu"
                )
                return arrow_molora_reference(
                    x=x,
                    base_out=base_out,
                    prototypes=prototypes,
                    a_weights=a_weights,
                    b_weights=b_weights,
                    ranks=ranks,
                    scalings=scalings,
                    top_k=top_k,
                    temperature=temperature,
                    layer_name=resolved_layer,
                    backend_label=dev_label,
                )
            target = "mps" if has_mps() else "cpu"
            label: Literal["torch_mps", "torch_cpu"] = (
                "torch_mps" if target == "mps" else "torch_cpu"
            )
            return self._run_torch_on_device(
                torch_mod=torch_mod,
                target_device=target,
                backend_label=label,
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        # backend == "auto"
        torch_mod = optional_torch()
        if torch_mod is None or not _is_torch_tensor(x):
            return self._run_pure_python(
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        if bool(x.is_cuda) and has_triton_cuda():
            return self._run_triton_cuda(
                x=x,
                base_out=base_out,
                prototypes=prototypes,
                a_weights=a_weights,
                b_weights=b_weights,
                ranks=ranks,
                scalings=scalings,
                top_k=top_k,
                temperature=temperature,
                layer_name=resolved_layer,
            )

        auto_label: Literal["torch_mps", "torch_cpu"] = (
            "torch_mps" if str(x.device.type).lower() == "mps" else "torch_cpu"
        )
        return arrow_molora_reference(
            x=x,
            base_out=base_out,
            prototypes=prototypes,
            a_weights=a_weights,
            b_weights=b_weights,
            ranks=ranks,
            scalings=scalings,
            top_k=top_k,
            temperature=temperature,
            layer_name=resolved_layer,
            backend_label=auto_label,
        )

    @staticmethod
    def _run_pure_python(
        *,
        x: Any,
        base_out: Any,
        prototypes: Any,
        a_weights: Any,
        b_weights: Any,
        ranks: Any,
        scalings: Any,
        top_k: int,
        temperature: float,
        layer_name: str,
    ) -> tuple[Any, MoLoRARouteTelemetryBatch]:
        def _to_py(val: Any) -> Any:
            return val.detach().cpu().tolist() if _is_torch_tensor(val) else val

        return pure_python_arrow_reference(
            x=_to_py(x),
            base_out=_to_py(base_out),
            prototypes=_to_py(prototypes),
            a_weights=_to_py(a_weights),
            b_weights=_to_py(b_weights),
            ranks=_to_py(ranks),
            scalings=_to_py(scalings),
            top_k=top_k,
            temperature=temperature,
            layer_name=layer_name,
        )

    @staticmethod
    def _run_torch_on_device(
        *,
        torch_mod: Any,
        target_device: str,
        backend_label: Literal["torch_mps", "torch_cpu"],
        x: Any,
        base_out: Any,
        prototypes: Any,
        a_weights: Any,
        b_weights: Any,
        ranks: Any,
        scalings: Any,
        top_k: int,
        temperature: float,
        layer_name: str,
    ) -> tuple[Any, MoLoRARouteTelemetryBatch]:
        def _move(val: Any, dtype: Any) -> Any:
            if isinstance(val, torch_mod.Tensor):
                return val.to(device=target_device)
            return torch_mod.tensor(val, dtype=dtype, device=target_device)

        x_t = _move(x, torch_mod.float32)
        base_t = _move(base_out, x_t.dtype)
        proto_t = _move(prototypes, torch_mod.float32)
        a_t = _move(a_weights, x_t.dtype)
        b_t = _move(b_weights, x_t.dtype)
        ranks_t = _move(ranks, torch_mod.int32)
        scalings_t = _move(scalings, torch_mod.float32)

        return arrow_molora_reference(
            x=x_t,
            base_out=base_t,
            prototypes=proto_t,
            a_weights=a_t,
            b_weights=b_t,
            ranks=ranks_t,
            scalings=scalings_t,
            top_k=top_k,
            temperature=temperature,
            layer_name=layer_name,
            backend_label=backend_label,
        )

    @staticmethod
    def _run_triton_cuda(
        *,
        x: Any,
        base_out: Any,
        prototypes: Any,
        a_weights: Any,
        b_weights: Any,
        ranks: Any,
        scalings: Any,
        top_k: int,
        temperature: float,
        layer_name: str,
    ) -> tuple[Any, MoLoRARouteTelemetryBatch]:
        torch_mod = optional_torch()
        assert torch_mod is not None
        if not isinstance(x, torch_mod.Tensor) or not bool(x.is_cuda):
            raise RuntimeExecutionError("triton_cuda dispatch requires CUDA PyTorch Tensor inputs.")
        if not isinstance(base_out, torch_mod.Tensor) or not bool(base_out.is_cuda):
            raise RuntimeExecutionError(
                "triton_cuda dispatch requires CUDA PyTorch Tensor for base_out."
            )
        if x.ndim not in (2, 3) or base_out.ndim != x.ndim:
            raise ConfigError(
                f"x and base_out must both be 2D or 3D tensors; got {tuple(x.shape)} and "
                f"{tuple(base_out.shape)}."
            )
        if tuple(x.shape[:-1]) != tuple(base_out.shape[:-1]):
            raise ConfigError("Leading shape mismatch between x and base_out.")

        d_in = int(a_weights.shape[2])
        num_experts = int(a_weights.shape[0])
        x_2d = x.reshape(-1, d_in)
        num_tokens = int(x_2d.shape[0])

        expert_ids, routing_weights, entropy = route_topk_tokens(
            x_2d, prototypes, top_k=top_k, temperature=temperature
        )
        if num_tokens == 0:
            telemetry = MoLoRARouteTelemetryBatch(
                layer_name=layer_name,
                num_tokens=0,
                top_k=top_k,
                expert_indices=[],
                routing_weights=[],
                router_entropy=[],
                backend="triton_cuda",
            )
            return base_out.clone(), telemetry

        layout = build_segment_metadata(expert_ids, routing_weights, num_experts)
        y_out = triton_arrow_sgmv_forward(
            x=x,
            base_out=base_out,
            a_packed=a_weights,
            b_packed=b_weights,
            ranks=ranks,
            scalings=scalings,
            segment_layout=layout,
        )
        telemetry = MoLoRARouteTelemetryBatch(
            layer_name=layer_name,
            num_tokens=num_tokens,
            top_k=top_k,
            expert_indices=cast(list[list[int]], expert_ids.detach().cpu().tolist()),
            routing_weights=cast(list[list[float]], routing_weights.detach().cpu().tolist()),
            router_entropy=cast(list[float], entropy.detach().cpu().tolist()),
            backend="triton_cuda",
        )
        return y_out, telemetry
