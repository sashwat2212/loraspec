"""Integration tests for zero-fork `vLLM V1` entry-point plugin & telemetry ring buffer."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

import pytest

from loraspec.core.errors import EXIT_CONFIG_ERROR, AdapterParseError, ConfigError
from loraspec.core.spectral_math import Matrix
from loraspec.inspect.safetensors_reader import load_peft_adapter, write_safetensors
from loraspec.kernels.router import extract_arrow_prototypes
from loraspec.serving.vllm_plugin import (
    TelemetryRingBuffer,
    capture_decode_step_telemetry,
    get_registered_plugin_state,
    register,
    reset_plugin_state,
    wrap_model_with_molora,
)


def _write_domain_adapter(
    root: Path,
    *,
    domain_idx: int,
    d_in: int = 8,
    d_out: int = 8,
    rank: int = 2,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": float(rank * 2),
        "lora_dropout": 0.0,
        "use_rslora": False,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    col = domain_idx * 2
    a_mat: Matrix = [[1.0 if j == (col + r) else 0.0 for j in range(d_in)] for r in range(rank)]
    b_mat: Matrix = [[0.5 if i == (col + c) else 0.0 for c in range(rank)] for i in range(d_out)]
    write_safetensors(
        root / "adapter_model.safetensors",
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": (a_mat, "F32"),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": (b_mat, "F32"),
        },
        metadata={"format": "pt"},
    )
    return root


def test_vllm_entry_point_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify `vllm.general_plugins` resolves without patching vLLM files."""
    reset_plugin_state()
    eps = importlib.metadata.entry_points(group="vllm.general_plugins")
    matched = [ep for ep in eps if ep.name == "loraspec_molora"]
    assert matched, "Missing vllm.general_plugins entry point 'loraspec_molora'"
    assert matched[0].value == "loraspec.serving.vllm_plugin:register"

    loaded_fn = matched[0].load()
    monkeypatch.delenv("LORASPEC_MOLORA_ENABLE", raising=False)
    monkeypatch.delenv("LORASPEC_ADAPTERS", raising=False)
    # No-op when disabled
    assert loaded_fn(force_reset=True) is None
    assert get_registered_plugin_state() is None

    # Base model mismatch fails deterministically before corrupt serving
    d1 = _write_domain_adapter(tmp_path / "sql_lora", domain_idx=0, base_model="Qwen/Qwen2.5-3B")
    d2 = _write_domain_adapter(
        tmp_path / "rust_lora", domain_idx=1, base_model="meta-llama/Llama-3"
    )
    monkeypatch.setenv("LORASPEC_MOLORA_ENABLE", "1")
    monkeypatch.setenv("LORASPEC_ADAPTERS", f"{d1},{d2}")
    with pytest.raises(AdapterParseError) as exc_m:
        loaded_fn(force_reset=True)
    assert exc_m.value.exit_code == EXIT_CONFIG_ERROR
    reset_plugin_state()


def test_vllm_worker_hook_mixed_domain_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify worker hook routes SQL, Rust, and JSON tokens in one batch."""
    reset_plugin_state()
    sql_dir = _write_domain_adapter(tmp_path / "sql-lora", domain_idx=0)
    rust_dir = _write_domain_adapter(tmp_path / "rust-lora", domain_idx=1)
    json_dir = _write_domain_adapter(tmp_path / "json-lora", domain_idx=2)

    manifest_file = tmp_path / "adapters.json"
    manifest_file.write_text(
        json.dumps([str(sql_dir), {"path": str(rust_dir)}, str(json_dir)]),
        encoding="utf-8",
    )

    monkeypatch.setenv("LORASPEC_MOLORA_ENABLE", "1")
    monkeypatch.setenv("LORASPEC_ADAPTERS", str(manifest_file))
    monkeypatch.setenv("LORASPEC_MOLORA_TOP_K", "2")
    state = register(force_reset=True)
    assert state is not None
    assert state.enabled is True
    assert state.adapter_ids == ("sql-lora", "rust-lora", "json-lora")

    class _BaseLinear:
        def __call__(self, x: Any) -> Any:
            return [[0.0] * 8 for _ in x]

    class _FakeBlock:
        def __init__(self) -> None:
            self.q_proj: Any = _BaseLinear()

        def named_children(self) -> list[tuple[str, Any]]:
            return [("q_proj", self.q_proj)]

    class _FakeModel:
        def __init__(self) -> None:
            self.layer0 = _FakeBlock()

        def named_children(self) -> list[tuple[str, Any]]:
            return [("layer0", self.layer0)]

    loaded = [load_peft_adapter(sql_dir), load_peft_adapter(rust_dir), load_peft_adapter(json_dir)]
    bank = extract_arrow_prototypes(loaded)
    model = _FakeModel()
    wrapped, ring = wrap_model_with_molora(model, bank, top_k=2, probe_layers=["layer0.q_proj"])
    assert wrapped == ["layer0.q_proj"]

    # Token 0 aligned with sql (dims 0..1), Token 1 with rust (dims 2..3), Token 2 with json (4..5)
    x_batch: Matrix = [
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
    ]
    out = model.layer0.q_proj(
        x_batch, token_texts=["SELECT", "pub fn", '{"ok": true}'], itl_ms=10.5
    )
    assert len(out) == 3
    frames = ring.snapshot()
    assert len(frames) == 3
    assert frames[0].top_k_adapters[0] == "sql-lora"
    assert frames[1].top_k_adapters[0] == "rust-lora"
    assert frames[2].top_k_adapters[0] == "json-lora"
    reset_plugin_state()


def test_telemetry_ring_buffer_bounded_nonblocking_capture() -> None:
    """Verify bounded ring buffer drops oldest frames cleanly with overflow counting."""
    with pytest.raises(ConfigError):
        TelemetryRingBuffer(capacity=0)

    ring = TelemetryRingBuffer(capacity=3)
    captured = capture_decode_step_telemetry(
        ring,
        step_start_index=10,
        token_texts=["tok0", "tok1", "tok2", "tok3", "tok4"],
        adapter_ids=["sql-lora", "rust-lora", "json-lora"],
        expert_indices=[[0, 1], [1, 2], [2, 0], [0, 2], [1, 0]],
        routing_weights=[[0.8, 0.2]] * 5,
        router_entropy=[0.5] * 5,
        itl_ms=[9.5, 10.0, 10.5, 11.0, 11.5],
    )
    assert len(captured) == 5
    assert len(ring) == 3
    assert ring.overflow_count == 2
    snap = ring.snapshot()
    assert [f.token_index for f in snap] == [12, 13, 14]
    drained = ring.drain(max_items=2)
    assert [f.token_text for f in drained] == ["tok2", "tok3"]
    assert len(ring) == 1
