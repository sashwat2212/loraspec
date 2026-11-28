"""Zero-fork `vLLM V1` entry-point plugin and bounded lock-free telemetry ring buffer."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loraspec.core.errors import AdapterParseError, ConfigError, RuntimeExecutionError
from loraspec.core.schemas import MoLoRARouteTelemetry
from loraspec.inspect.safetensors_reader import LoadedLoRAWeights, load_peft_adapter
from loraspec.kernels import has_triton_cuda, optional_torch
from loraspec.kernels.dispatch import ArrowMoLoRADispatcher, DispatchBackend
from loraspec.kernels.router import ArrowPrototypeBank, extract_arrow_prototypes

TARGET_LINEAR_SUFFIXES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TelemetryRingBuffer:
    """Bounded lock-protected ring buffer recording decode-step Top-k routing frames."""

    def __init__(self, capacity: int = 4096) -> None:
        if isinstance(capacity, bool) or capacity < 1:
            raise ConfigError(f"TelemetryRingBuffer capacity must be >= 1, got {capacity!r}.")
        self._capacity = int(capacity)
        self._buf: deque[MoLoRARouteTelemetry] = deque()
        self._overflow_count = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def overflow_count(self) -> int:
        with self._lock:
            return self._overflow_count

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def push(self, frame: MoLoRARouteTelemetry) -> None:
        with self._lock:
            if len(self._buf) >= self._capacity:
                self._buf.popleft()
                self._overflow_count += 1
            self._buf.append(frame)

    def extend(self, frames: Sequence[MoLoRARouteTelemetry]) -> None:
        with self._lock:
            for frame in frames:
                if len(self._buf) >= self._capacity:
                    self._buf.popleft()
                    self._overflow_count += 1
                self._buf.append(frame)

    def snapshot(self) -> list[MoLoRARouteTelemetry]:
        with self._lock:
            return list(self._buf)

    def drain(self, max_items: int | None = None) -> list[MoLoRARouteTelemetry]:
        if max_items is not None and max_items < 0:
            raise ConfigError(f"max_items must be >= 0, got {max_items!r}.")
        with self._lock:
            if max_items is None or max_items >= len(self._buf):
                items = list(self._buf)
                self._buf.clear()
                return items
            items = [self._buf.popleft() for _ in range(max_items)]
            return items

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            self._overflow_count = 0


def _to_python_seq(value: Any) -> Any:
    torch_mod = optional_torch()
    if torch_mod is not None and isinstance(value, torch_mod.Tensor):
        return value.detach().cpu().tolist()
    return value


def capture_decode_step_telemetry(
    ring_buffer: TelemetryRingBuffer,
    *,
    step_start_index: int,
    token_texts: Sequence[str],
    adapter_ids: Sequence[str],
    expert_indices: Any,
    routing_weights: Any,
    router_entropy: Any,
    itl_ms: float | Sequence[float] = 0.0,
) -> list[MoLoRARouteTelemetry]:
    """Translate step-boundary routing tensors into validated frames in `TelemetryRingBuffer`."""
    exp_rows = cast(Sequence[Sequence[int]], _to_python_seq(expert_indices))
    w_rows = cast(Sequence[Sequence[float]], _to_python_seq(routing_weights))
    ent_list = cast(Sequence[float], _to_python_seq(router_entropy))

    num_tokens = len(exp_rows)
    if len(token_texts) != num_tokens or len(w_rows) != num_tokens or len(ent_list) != num_tokens:
        raise ConfigError(
            "Length mismatch in capture_decode_step_telemetry: "
            f"tokens={len(token_texts)}, experts={num_tokens}, weights={len(w_rows)}, "
            f"entropy={len(ent_list)}."
        )
    if isinstance(itl_ms, Sequence):
        if len(itl_ms) != num_tokens:
            raise ConfigError(f"itl_ms sequence length {len(itl_ms)} != {num_tokens}.")
        itl_values = [float(v) for v in itl_ms]
    else:
        itl_values = [float(itl_ms)] * num_tokens

    frames: list[MoLoRARouteTelemetry] = []
    num_adapters = len(adapter_ids)
    for offset in range(num_tokens):
        chosen_names: list[str] = []
        for idx_raw in exp_rows[offset]:
            idx = int(idx_raw)
            chosen_names.append(adapter_ids[idx] if 0 <= idx < num_adapters else f"expert_{idx}")
        frame = MoLoRARouteTelemetry(
            token_index=int(step_start_index) + offset,
            token_text=str(token_texts[offset]),
            top_k_adapters=chosen_names,
            top_k_weights=[round(float(w), 6) for w in w_rows[offset]],
            router_entropy=round(float(ent_list[offset]), 6),
            itl_ms=round(itl_values[offset], 4),
        )
        frames.append(frame)

    ring_buffer.extend(frames)
    return frames


class MoLoRALinearAdapter:
    """Drop-in linear wrapper adding fused Top-k Arrow-MoLoRA delta over native `W_0 x`."""

    def __init__(
        self,
        base_layer: Any,
        *,
        adapter_ids: Sequence[str],
        prototypes: Any,
        a_weights: Any,
        b_weights: Any,
        ranks: Any,
        scalings: Any,
        top_k: int = 2,
        temperature: float = 1.0,
        layer_name: str = "molora_layer",
        capture_probe: bool = False,
        dispatcher: ArrowMoLoRADispatcher | None = None,
        ring_buffer: TelemetryRingBuffer | None = None,
        backend: DispatchBackend = "auto",
    ) -> None:
        if top_k < 1:
            raise ConfigError(f"MoLoRALinearAdapter top_k must be >= 1, got {top_k!r}.")
        if temperature <= 0.0:
            raise ConfigError(f"MoLoRALinearAdapter temperature must be > 0, got {temperature!r}.")
        self.base_layer = base_layer
        self.adapter_ids = tuple(adapter_ids)
        self.prototypes = prototypes
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.ranks = ranks
        self.scalings = scalings
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.layer_name = layer_name
        self.capture_probe = bool(capture_probe)
        self.dispatcher = dispatcher or ArrowMoLoRADispatcher()
        self.ring_buffer = ring_buffer
        self.backend: DispatchBackend = backend
        self._token_cursor = 0

    def forward(
        self,
        x: Any,
        *,
        token_texts: Sequence[str] | None = None,
        itl_ms: float = 0.0,
        backend: DispatchBackend | None = None,
    ) -> Any:
        base_out = self.base_layer(x)
        resolved_backend = backend or self.backend
        out, batch = self.dispatcher.forward(
            x=x,
            base_out=base_out,
            prototypes=self.prototypes,
            a_weights=self.a_weights,
            b_weights=self.b_weights,
            ranks=self.ranks,
            scalings=self.scalings,
            top_k=self.top_k,
            temperature=self.temperature,
            backend=resolved_backend,
            layer_name=self.layer_name,
            include_telemetry=self.capture_probe,
        )
        if self.capture_probe and self.ring_buffer is not None and batch.num_tokens > 0:
            texts = (
                list(token_texts)
                if token_texts is not None
                else [f"<tok_{self._token_cursor + i}>" for i in range(batch.num_tokens)]
            )
            capture_decode_step_telemetry(
                self.ring_buffer,
                step_start_index=self._token_cursor,
                token_texts=texts,
                adapter_ids=self.adapter_ids,
                expert_indices=batch.expert_indices,
                routing_weights=batch.routing_weights,
                router_entropy=batch.router_entropy,
                itl_ms=itl_ms,
            )
            self._token_cursor += batch.num_tokens
        return out

    def __call__(
        self,
        x: Any,
        *,
        token_texts: Sequence[str] | None = None,
        itl_ms: float = 0.0,
        backend: DispatchBackend | None = None,
    ) -> Any:
        return self.forward(x, token_texts=token_texts, itl_ms=itl_ms, backend=backend)


def _match_bank_layer(full_attr_path: str, available_layers: Sequence[str]) -> str | None:
    if full_attr_path in available_layers:
        return full_attr_path
    suffix_matches = [
        cand
        for cand in available_layers
        if cand == full_attr_path or cand.endswith(f".{full_attr_path}")
    ]
    if suffix_matches:
        return sorted(suffix_matches)[0]
    short_leaf = full_attr_path.rsplit(".", maxsplit=1)[-1]
    if short_leaf in TARGET_LINEAR_SUFFIXES:
        leaf_matches = [cand for cand in available_layers if cand.endswith(f".{short_leaf}")]
        if leaf_matches:
            return sorted(leaf_matches)[0]
    return None


def wrap_model_with_molora(
    model: Any,
    bank: ArrowPrototypeBank,
    *,
    top_k: int = 2,
    temperature: float = 1.0,
    ring_buffer: TelemetryRingBuffer | None = None,
    probe_layers: Sequence[str] | None = None,
    backend: DispatchBackend = "auto",
    device: Any = None,
    dtype: Any = None,
) -> tuple[list[str], TelemetryRingBuffer]:
    """Attach `MoLoRALinearAdapter` wrappers to matching linear projections on `model`."""
    buf = ring_buffer or TelemetryRingBuffer()
    wrapped: list[str] = []
    probe_set = set(probe_layers) if probe_layers is not None else None
    use_tensors = optional_torch() is not None and device is not None

    def _visit(module: Any, prefix: str) -> None:
        named_children = getattr(module, "named_children", None)
        if callable(named_children):
            children = list(cast(Any, named_children()))
            for child_name, child_mod in children:
                child_path = f"{prefix}.{child_name}" if prefix else str(child_name)
                if child_name.endswith(TARGET_LINEAR_SUFFIXES) and not isinstance(
                    child_mod, MoLoRALinearAdapter
                ):
                    matched_layer = _match_bank_layer(child_path, bank.layer_names)
                    if matched_layer is not None:
                        capture_flag = (
                            (child_path in probe_set or matched_layer in probe_set)
                            if probe_set is not None
                            else len(wrapped) == 0
                        )
                        if use_tensors:
                            p_w, a_w, b_w, r_w, s_w = bank.layer_tensors(
                                matched_layer, device=device, dtype=dtype
                            )
                        else:
                            p_w = bank.prototypes[matched_layer]
                            a_w = bank.a_packed[matched_layer]
                            b_w = bank.b_packed[matched_layer]
                            r_w = bank.ranks[matched_layer]
                            s_w = bank.scalings[matched_layer]
                        wrapper = MoLoRALinearAdapter(
                            child_mod,
                            adapter_ids=bank.adapter_ids,
                            prototypes=p_w,
                            a_weights=a_w,
                            b_weights=b_w,
                            ranks=r_w,
                            scalings=s_w,
                            top_k=top_k,
                            temperature=temperature,
                            layer_name=matched_layer,
                            capture_probe=capture_flag,
                            ring_buffer=buf,
                            backend=backend,
                        )
                        setattr(module, child_name, wrapper)
                        wrapped.append(child_path)
                        continue
                _visit(child_mod, child_path)
            return

        if isinstance(module, dict):
            mod_dict = cast(dict[str, Any], module)
            for key, child_obj in list(mod_dict.items()):
                child_path = f"{prefix}.{key}" if prefix else key
                if callable(child_obj) and key.endswith(TARGET_LINEAR_SUFFIXES):
                    matched_layer = _match_bank_layer(child_path, bank.layer_names)
                    if matched_layer is not None:
                        capture_flag = (
                            (child_path in probe_set or matched_layer in probe_set)
                            if probe_set is not None
                            else len(wrapped) == 0
                        )
                        if use_tensors:
                            p_w, a_w, b_w, r_w, s_w = bank.layer_tensors(
                                matched_layer, device=device, dtype=dtype
                            )
                        else:
                            p_w = bank.prototypes[matched_layer]
                            a_w = bank.a_packed[matched_layer]
                            b_w = bank.b_packed[matched_layer]
                            r_w = bank.ranks[matched_layer]
                            s_w = bank.scalings[matched_layer]
                        mod_dict[key] = MoLoRALinearAdapter(
                            child_obj,
                            adapter_ids=bank.adapter_ids,
                            prototypes=p_w,
                            a_weights=a_w,
                            b_weights=b_w,
                            ranks=r_w,
                            scalings=s_w,
                            top_k=top_k,
                            temperature=temperature,
                            layer_name=matched_layer,
                            capture_probe=capture_flag,
                            ring_buffer=buf,
                            backend=backend,
                        )
                        wrapped.append(child_path)

    _visit(model, "")
    return wrapped, buf


@dataclass(frozen=True)
class RegisteredPluginState:
    """Resolved runtime descriptor created by `loraspec.serving.vllm_plugin:register`."""

    enabled: bool
    adapter_ids: tuple[str, ...]
    bank: ArrowPrototypeBank
    ring_buffer: TelemetryRingBuffer
    top_k: int
    temperature: float
    backend: str
    vllm_hook_attached: bool


_active_plugin_state: RegisteredPluginState | None = None


def get_registered_plugin_state() -> RegisteredPluginState | None:
    return _active_plugin_state


def reset_plugin_state() -> None:
    global _active_plugin_state
    _active_plugin_state = None


def _parse_adapter_env(raw_adapters: str) -> list[Path]:
    text = raw_adapters.strip()
    if not text:
        raise ConfigError(
            "LORASPEC_MOLORA_ENABLE=1 requires LORASPEC_ADAPTERS to list >= 1 adapter path."
        )
    candidate_file = Path(text)
    if candidate_file.is_file():
        try:
            loaded_json: object = json.loads(candidate_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(
                f"Failed to read LORASPEC_ADAPTERS manifest '{candidate_file}': {exc}"
            ) from exc
        if not isinstance(loaded_json, list):
            raise ConfigError("LORASPEC_ADAPTERS JSON manifest must be a list of paths.")
        resolved_paths: list[Path] = []
        for entry in cast(list[object], loaded_json):
            if isinstance(entry, str):
                resolved_paths.append(Path(entry))
            elif isinstance(entry, dict) and isinstance(
                cast(dict[str, object], entry).get("path"), str
            ):
                resolved_paths.append(Path(cast(str, cast(dict[str, object], entry)["path"])))
            else:
                raise ConfigError(f"Unsupported entry in LORASPEC_ADAPTERS manifest: {entry!r}.")
        if not resolved_paths:
            raise ConfigError("LORASPEC_ADAPTERS manifest contains zero adapter paths.")
        return resolved_paths

    if text.startswith("["):
        try:
            inline_json: object = json.loads(text)
        except ValueError as exc:
            raise ConfigError(f"Malformed JSON in LORASPEC_ADAPTERS: {exc}") from exc
        if not isinstance(inline_json, list) or not inline_json:
            raise ConfigError("LORASPEC_ADAPTERS JSON array must be non-empty.")
        return [Path(str(item)) for item in cast(list[object], inline_json)]

    parts = [Path(p.strip()) for p in text.split(",") if p.strip()]
    if not parts:
        raise ConfigError("LORASPEC_ADAPTERS must contain at least one adapter path.")
    return parts


def register(*, force_reset: bool = False) -> RegisteredPluginState | None:
    """Entry point for `vllm.general_plugins` (`loraspec_molora`). No-op unless enabled."""
    global _active_plugin_state
    if force_reset:
        _active_plugin_state = None

    if os.environ.get("LORASPEC_MOLORA_ENABLE", "0").strip() != "1":
        return None

    adapter_paths = _parse_adapter_env(os.environ.get("LORASPEC_ADAPTERS", ""))
    top_k_raw = int(os.environ.get("LORASPEC_MOLORA_TOP_K", "2"))
    temp_raw = float(os.environ.get("LORASPEC_MOLORA_TEMPERATURE", "1.0"))
    ring_cap = int(os.environ.get("LORASPEC_TELEMETRY_CAPACITY", "4096"))
    if top_k_raw < 1:
        raise ConfigError(f"LORASPEC_MOLORA_TOP_K must be >= 1, got {top_k_raw}.")
    if temp_raw <= 0.0:
        raise ConfigError(f"LORASPEC_MOLORA_TEMPERATURE must be > 0, got {temp_raw}.")

    loaded: list[LoadedLoRAWeights] = [load_peft_adapter(p) for p in adapter_paths]
    if len(loaded) < top_k_raw:
        top_k_raw = len(loaded)

    try:
        bank = extract_arrow_prototypes(loaded, layer_policy="union", backend="auto")
    except AdapterParseError:
        raise
    except Exception as exc:
        raise RuntimeExecutionError(
            f"Failed to initialize ArrowPrototypeBank for vLLM plugin: {exc}"
        ) from exc

    ring = TelemetryRingBuffer(capacity=ring_cap)
    vllm_attached = False
    if importlib.util.find_spec("vllm") is not None:
        try:
            _ = importlib.import_module("vllm")
            vllm_attached = True
        except Exception:
            vllm_attached = False

    backend_name = "triton_cuda" if has_triton_cuda() else "torch_reference"
    state = RegisteredPluginState(
        enabled=True,
        adapter_ids=bank.adapter_ids,
        bank=bank,
        ring_buffer=ring,
        top_k=top_k_raw,
        temperature=temp_raw,
        backend=backend_name,
        vllm_hook_attached=vllm_attached,
    )
    _active_plugin_state = state
    return state
