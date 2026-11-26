"""Shared fixtures creating real binary `.safetensors` + PEFT LoRA test adapters."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from loraspec.inspect.safetensors_reader import write_safetensors

LAYER_PREFIXES: tuple[str, ...] = (
    "base_model.model.layers.0.self_attn.q_proj",
    "base_model.model.layers.0.mlp.gate_proj",
    "base_model.model.layers.0.mlp.up_proj",
    "base_model.model.layers.0.mlp.down_proj",
)


def _identity_block(
    rows: int, cols: int, row_offset: int = 0, scale: float = 1.0
) -> list[list[float]]:
    mat = [[0.0] * cols for _ in range(rows)]
    for c in range(cols):
        r = row_offset + c
        if 0 <= r < rows:
            mat[r][c] = scale
    return mat


def _write_peft_adapter(
    root: Path,
    *,
    base_model: str = "Qwen/Qwen2.5-3B-Instruct",
    rank: int = 4,
    alpha: float = 4.0,
    use_rslora: bool = False,
    target_modules: list[str] | None = None,
    tensors: dict[str, tuple[list[list[float]], str]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    targets = target_modules or ["q_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "use_rslora": use_rslora,
        "bias": "none",
        "target_modules": targets,
    }
    (root / "adapter_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_safetensors(root / "adapter_model.safetensors", tensors, metadata={"format": "pt"})
    return root


@pytest.fixture()
def orthogonal_adapter_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two clean all-linear rank-4 adapters occupying disjoint orthogonal 4D subspaces."""
    d = 16
    r = 4

    tensors_a: dict[str, tuple[list[list[float]], str]] = {}
    tensors_b: dict[str, tuple[list[list[float]], str]] = {}
    for prefix in LAYER_PREFIXES:
        # Adapter A occupies basis rows/cols [0..4)
        a_mat = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
        b_mat = _identity_block(d, r, row_offset=0, scale=1.0)
        tensors_a[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors_a[f"{prefix}.lora_B.weight"] = (b_mat, "F32")

        # Adapter B occupies orthogonal basis rows/cols [4..8)
        a_mat_b = [[1.0 if j == (i + r) else 0.0 for j in range(d)] for i in range(r)]
        b_mat_b = _identity_block(d, r, row_offset=r, scale=1.0)
        tensors_b[f"{prefix}.lora_A.weight"] = (a_mat_b, "F32")
        tensors_b[f"{prefix}.lora_B.weight"] = (b_mat_b, "F32")

    path_a = _write_peft_adapter(tmp_path / "sql_lora", rank=r, alpha=float(r), tensors=tensors_a)
    path_b = _write_peft_adapter(tmp_path / "rust_lora", rank=r, alpha=float(r), tensors=tensors_b)
    return path_a, path_b


@pytest.fixture()
def colliding_conflicting_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two rank-4 adapters occupying the exact same subspace with opposite signs."""
    d = 16
    r = 4
    tensors_a: dict[str, tuple[list[list[float]], str]] = {}
    tensors_b: dict[str, tuple[list[list[float]], str]] = {}
    for prefix in LAYER_PREFIXES:
        a_mat = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
        b_pos = _identity_block(d, r, row_offset=0, scale=1.5)
        b_neg = _identity_block(d, r, row_offset=0, scale=-1.5)
        tensors_a[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors_a[f"{prefix}.lora_B.weight"] = (b_pos, "F32")
        tensors_b[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors_b[f"{prefix}.lora_B.weight"] = (b_neg, "F32")

    path_a = _write_peft_adapter(tmp_path / "coll_a", rank=r, alpha=float(r), tensors=tensors_a)
    path_b = _write_peft_adapter(tmp_path / "coll_b", rank=r, alpha=float(r), tensors=tensors_b)
    return path_a, path_b


@pytest.fixture()
def rank_collapsed_adapter(tmp_path: Path) -> Path:
    """Rank-8 adapter whose energy is 100% concentrated in a single rank-1 direction."""
    d = 16
    r = 8
    tensors: dict[str, tuple[list[list[float]], str]] = {}
    for prefix in LAYER_PREFIXES:
        a_mat = [[1.0 if (i == 0 and j == 0) else 0.0 for j in range(d)] for i in range(r)]
        b_mat = [[4.0 if (row == 0 and col == 0) else 0.0 for col in range(r)] for row in range(d)]
        tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")

    return _write_peft_adapter(tmp_path / "collapsed_lora", rank=r, alpha=float(r), tensors=tensors)


@pytest.fixture()
def spiked_attention_only_adapter(tmp_path: Path) -> Path:
    """Attention-only adapter with a 20x spectral norm spike on layer 3."""
    d = 16
    r = 4
    attn_layers = [
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.1.self_attn.q_proj",
        "base_model.model.layers.2.self_attn.q_proj",
        "base_model.model.layers.3.self_attn.q_proj",
    ]
    tensors: dict[str, tuple[list[list[float]], str]] = {}
    for idx, prefix in enumerate(attn_layers):
        scale = 25.0 if idx == 3 else 1.0
        a_mat = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
        b_mat = _identity_block(d, r, row_offset=0, scale=scale)
        tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")

    return _write_peft_adapter(
        tmp_path / "spiked_attn_lora",
        rank=r,
        alpha=float(r),
        target_modules=["q_proj"],
        tensors=tensors,
    )


@pytest.fixture()
def dead_nan_adapter(tmp_path: Path) -> Path:
    """Corrupted adapter containing NaN tensor values."""
    d = 8
    r = 2
    tensors: dict[str, tuple[list[list[float]], str]] = {}
    for idx, prefix in enumerate(LAYER_PREFIXES):
        a_mat = [[1.0 if j == i else 0.0 for j in range(d)] for i in range(r)]
        b_mat = _identity_block(d, r, row_offset=0, scale=1.0)
        if idx == 0:
            b_mat[0][0] = math.nan
        tensors[f"{prefix}.lora_A.weight"] = (a_mat, "F32")
        tensors[f"{prefix}.lora_B.weight"] = (b_mat, "F32")

    return _write_peft_adapter(tmp_path / "nan_lora", rank=r, alpha=float(r), tensors=tensors)
