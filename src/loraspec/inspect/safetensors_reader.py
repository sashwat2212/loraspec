"""Bounds-safe, pickle-free `.safetensors` binary reader/writer and PEFT adapter loader."""

from __future__ import annotations

import array
import importlib
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loraspec.core.errors import AdapterParseError
from loraspec.core.manifest import sha256_file
from loraspec.core.schemas import AdapterFormat, LoRAAdapterSpec, LoRALayerSpec
from loraspec.core.spectral_math import MathBackend, validate_math_backend

MAX_HEADER_BYTES: int = 100 * 1024 * 1024  # 100 MiB ceiling per Threat T2

DTYPE_WIDTH_BYTES: dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

FLOAT_DTYPES: set[str] = {"F32", "F16", "BF16", "F64"}

_LORA_KEY_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<side>lora_A|lora_B)(?:\.(?P<adapter>[^.]+))?\.weight$"
)
_DORA_MAG_KEY_RE = re.compile(
    r"^(?:(?P<prefix>.+)\.)?lora_magnitude_vector"
    r"(?:\.(?P<adapter>(?!weight$)[^.]+))?(?:\.weight)?$"
)
_REGEX_META_CHARS = frozenset("^$*+?[]()|\\")


def strip_peft_base_prefix(layer_name: str) -> str:
    """Strip PEFT wrapper prefix (`base_model.model.` or `base_model.`) for pattern matching."""
    if layer_name.startswith("base_model.model."):
        stripped = layer_name.removeprefix("base_model.model.")
        return stripped or layer_name
    if layer_name.startswith("base_model."):
        stripped = layer_name.removeprefix("base_model.")
        return stripped or layer_name
    return layer_name


def match_peft_pattern(
    layer_name: str,
    pattern: dict[str, Any],
    *,
    config_path: Path | None = None,
) -> tuple[bool, Any]:
    """Match a canonical LoRA layer name against PEFT `rank_pattern` / `alpha_pattern` keys."""
    if not pattern:
        return False, None
    if layer_name in pattern:
        return True, pattern[layer_name]
    if layer_name == "__root__" and "" in pattern:
        return True, pattern[""]

    stripped = strip_peft_base_prefix(layer_name)
    if stripped in pattern:
        return True, pattern[stripped]

    suffix_matches: list[tuple[int, int, str, Any]] = []
    for key, val in pattern.items():
        if not key:
            continue
        if stripped.endswith(f".{key}"):
            suffix_matches.append((len(key), 1, key, val))
        elif layer_name.endswith(f".{key}"):
            suffix_matches.append((len(key), 0, key, val))
    if suffix_matches:
        suffix_matches.sort(reverse=True)
        return True, suffix_matches[0][3]

    regex_matches: list[tuple[int, str, Any]] = []
    for key, val in pattern.items():
        if not key or not any(ch in _REGEX_META_CHARS for ch in key):
            continue
        try:
            if re.fullmatch(key, stripped) is not None or re.fullmatch(key, layer_name) is not None:
                regex_matches.append((len(key), key, val))
        except re.error as exc:
            raise AdapterParseError(
                f"Invalid regex key '{key}' in PEFT pattern ({config_path}): {exc}"
            ) from exc
    if regex_matches:
        regex_matches.sort(reverse=True)
        return True, regex_matches[0][2]

    return False, None


def get_layer_scaling(spec: LoRAAdapterSpec, layer_name: str) -> float:
    """Return effective per-layer scaling (`lora_alpha / r` or `/ sqrt(r)`), falling back safely."""
    for layer in spec.layers:
        if layer.layer_name == layer_name and layer.scaling is not None:
            return layer.scaling
    return spec.scaling


def _optional_numpy() -> Any | None:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _is_int_strict(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class SafetensorsTensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


@dataclass(frozen=True)
class LoadedSafetensors:
    path: Path
    metadata: dict[str, str]
    tensors: dict[str, SafetensorsTensorMeta]
    payload_offset: int
    payload_length: int


@dataclass(frozen=True)
class LoadedLoRAWeights:
    spec: LoRAAdapterSpec
    layer_factors: dict[str, tuple[list[list[float]], list[list[float]]]]


def _decode_bf16_buffer(raw: bytes) -> list[float]:
    count = len(raw) // 2
    if count == 0:
        return []
    if sys.byteorder == "little":
        u16_arr = array.array("H")
        u16_arr.frombytes(raw)
        u32_arr = array.array("I", (u << 16 for u in u16_arr))
        f32_arr = array.array("f")
        f32_arr.frombytes(u32_arr.tobytes())
        return f32_arr.tolist()
    values: list[float] = []
    for i in range(count):
        (u16,) = struct.unpack_from("<H", raw, i * 2)
        packed = struct.pack("<I", u16 << 16)
        (val,) = struct.unpack("<f", packed)
        values.append(float(val))
    return values


def _encode_bf16(values: list[float]) -> bytes:
    buf = bytearray(len(values) * 2)
    for idx, val in enumerate(values):
        raw_f32 = struct.pack("<f", float(val))
        (u32,) = struct.unpack("<I", raw_f32)
        u16 = (u32 >> 16) & 0xFFFF
        struct.pack_into("<H", buf, idx * 2, u16)
    return bytes(buf)


def decode_tensor_2d(
    path: Path,
    meta: SafetensorsTensorMeta,
    payload_offset: int,
    *,
    backend: MathBackend = "auto",
) -> list[list[float]]:
    """Decode a 2D tensor slice from a `.safetensors` file without `pickle`."""
    backend = validate_math_backend(backend)
    if len(meta.shape) != 2:
        raise AdapterParseError(
            f"Expected 2D LoRA tensor '{meta.name}' in {path}, got shape {meta.shape}."
        )
    if meta.dtype not in FLOAT_DTYPES:
        raise AdapterParseError(
            f"Unsupported dtype '{meta.dtype}' for LoRA tensor '{meta.name}' in {path}."
        )

    rows, cols = meta.shape
    begin, end = meta.data_offsets
    n_bytes = end - begin
    with path.open("rb") as fh:
        fh.seek(payload_offset + begin)
        raw = fh.read(n_bytes)
    if len(raw) != n_bytes:
        raise AdapterParseError(f"Truncated tensor payload for '{meta.name}' in {path}.")

    np_mod = _optional_numpy() if backend in ("auto", "numpy") else None
    if np_mod is not None and rows > 0 and cols > 0:
        if meta.dtype == "F32":
            arr = np_mod.frombuffer(raw, dtype="<f4").astype(np_mod.float64)
        elif meta.dtype == "F16":
            arr = np_mod.frombuffer(raw, dtype="<f2").astype(np_mod.float64)
        elif meta.dtype == "F64":
            arr = np_mod.frombuffer(raw, dtype="<f8")
        else:
            u32 = np_mod.frombuffer(raw, dtype="<u2").astype(np_mod.uint32) << 16
            arr = u32.view("<f4").astype(np_mod.float64)
        if int(arr.size) != rows * cols:
            raise AdapterParseError(
                f"Decoded element count mismatch for '{meta.name}': "
                f"expected {rows * cols}, got {int(arr.size)}."
            )
        reshaped: Any = arr.reshape((rows, cols))
        return cast(list[list[float]], reshaped.tolist())

    flat: list[float]
    if meta.dtype == "F32":
        flat = list(struct.unpack(f"<{len(raw) // 4}f", raw)) if raw else []
    elif meta.dtype == "F16":
        flat = list(struct.unpack(f"<{len(raw) // 2}e", raw)) if raw else []
    elif meta.dtype == "F64":
        flat = list(struct.unpack(f"<{len(raw) // 8}d", raw)) if raw else []
    elif meta.dtype == "BF16":
        flat = _decode_bf16_buffer(raw)
    else:
        raise AdapterParseError(f"Unsupported dtype '{meta.dtype}' in {path}.")

    if len(flat) != rows * cols:
        raise AdapterParseError(
            f"Decoded element count mismatch for '{meta.name}': "
            f"expected {rows * cols}, got {len(flat)}."
        )
    return [flat[r * cols : (r + 1) * cols] for r in range(rows)]


def parse_safetensors_header(path: Path) -> LoadedSafetensors:
    """Validate and parse a `.safetensors` header with strict offset bounds checking."""
    if not path.is_file():
        raise AdapterParseError(f"Safetensors file not found: {path}")

    file_size = path.stat().st_size
    if file_size < 8:
        raise AdapterParseError(f"Malformed .safetensors file (size < 8 bytes): {path}")

    with path.open("rb") as fh:
        header_len_bytes = fh.read(8)
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        if header_len == 0 or header_len > MAX_HEADER_BYTES:
            raise AdapterParseError(
                f"Unsafe or invalid .safetensors header length ({header_len} bytes) in {path}."
            )
        if 8 + header_len > file_size:
            raise AdapterParseError(
                f"Truncated .safetensors file: header exceeds file size ({file_size} bytes): {path}"
            )
        header_raw = fh.read(header_len)

    try:
        header_obj: Any = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterParseError(f"Invalid UTF-8 JSON header in {path}: {exc}") from exc

    if not isinstance(header_obj, dict):
        raise AdapterParseError(f"Top-level .safetensors header must be a JSON object: {path}")
    header_map = cast(dict[str, Any], header_obj)

    payload_offset = 8 + header_len
    payload_length = file_size - payload_offset
    metadata: dict[str, str] = {}
    tensors: dict[str, SafetensorsTensorMeta] = {}

    for key, val in header_map.items():
        if key == "__metadata__":
            if val is not None and not isinstance(val, dict):
                raise AdapterParseError(
                    f"Malformed __metadata__ entry in {path}: expected JSON object."
                )
            if isinstance(val, dict):
                meta_map = cast(dict[Any, Any], val)
                metadata = {str(mk): str(mv) for mk, mv in meta_map.items()}
            continue

        if not isinstance(val, dict):
            raise AdapterParseError(f"Malformed tensor entry '{key}' in {path}")
        entry = cast(dict[str, Any], val)

        dtype: Any = entry.get("dtype")
        shape_raw: Any = entry.get("shape")
        offsets_raw: Any = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or dtype not in DTYPE_WIDTH_BYTES
            or not isinstance(shape_raw, list)
            or not isinstance(offsets_raw, list)
        ):
            raise AdapterParseError(f"Malformed metadata fields for tensor '{key}' in {path}")
        shape_list = cast(list[Any], shape_raw)
        offsets_list = cast(list[Any], offsets_raw)
        if len(offsets_list) != 2:
            raise AdapterParseError(f"Malformed metadata fields for tensor '{key}' in {path}")

        shape: list[int] = []
        for dim in shape_list:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise AdapterParseError(f"Invalid negative/non-int shape in '{key}': {path}")
            shape.append(dim)

        begin: Any = offsets_list[0]
        end: Any = offsets_list[1]
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or isinstance(begin, bool)
            or isinstance(end, bool)
            or begin < 0
            or end < begin
            or end > payload_length
        ):
            raise AdapterParseError(
                f"Out-of-bounds data_offsets [{begin}, {end}] for '{key}' "
                f"(payload={payload_length}) in {path}"
            )

        expected_bytes = math.prod(shape) * DTYPE_WIDTH_BYTES[dtype]
        if end - begin != expected_bytes:
            raise AdapterParseError(
                f"Byte-size mismatch for tensor '{key}': offsets span {end - begin} bytes, "
                f"shape*dtype requires {expected_bytes} bytes in {path}"
            )

        tensors[key] = SafetensorsTensorMeta(
            name=key,
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(begin, end),
        )

    # Reject overlapping non-empty tensor byte slices (safetensors spec requirement)
    non_empty_spans = sorted(
        (t.data_offsets[0], t.data_offsets[1], t.name)
        for t in tensors.values()
        if t.data_offsets[1] > t.data_offsets[0]
    )
    for idx in range(1, len(non_empty_spans)):
        prev_begin, prev_end, prev_name = non_empty_spans[idx - 1]
        cur_begin, cur_end, cur_name = non_empty_spans[idx]
        if cur_begin < prev_end:
            raise AdapterParseError(
                f"Overlapping data_offsets between '{prev_name}' [{prev_begin}, {prev_end}] "
                f"and '{cur_name}' [{cur_begin}, {cur_end}] in {path}"
            )

    return LoadedSafetensors(
        path=path,
        metadata=metadata,
        tensors=tensors,
        payload_offset=payload_offset,
        payload_length=payload_length,
    )


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[list[list[float]], str]],
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write 2D float tensors (`F32`, `F16`, `BF16`) into a valid `.safetensors` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header_dict: dict[str, Any] = {}
    if metadata is not None:
        header_dict["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    payload_chunks: list[bytes] = []
    cursor = 0
    for name in sorted(tensors):
        matrix, dtype = tensors[name]
        if dtype not in {"F32", "F16", "BF16"}:
            raise AdapterParseError(f"Unsupported write dtype '{dtype}' for '{name}'.")
        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        flat: list[float] = []
        for row in matrix:
            if len(row) != cols:
                raise AdapterParseError(f"Ragged row length in tensor '{name}'.")
            flat.extend(float(x) for x in row)

        if dtype == "F32":
            raw = struct.pack(f"<{len(flat)}f", *flat) if flat else b""
        elif dtype == "F16":
            raw = struct.pack(f"<{len(flat)}e", *flat) if flat else b""
        else:
            raw = _encode_bf16(flat)

        end = cursor + len(raw)
        header_dict[name] = {
            "dtype": dtype,
            "shape": [rows, cols],
            "data_offsets": [cursor, end],
        }
        payload_chunks.append(raw)
        cursor = end

    header_bytes = json.dumps(header_dict, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(header_bytes)))
        fh.write(header_bytes)
        for chunk in payload_chunks:
            fh.write(chunk)
    return path


def _infer_module_type(layer_name: str) -> str:
    leaf = layer_name.rsplit(".", maxsplit=1)[-1]
    return leaf or layer_name


def _resolve_adapter_paths(path: Path) -> tuple[Path, Path, Path | None, AdapterFormat]:
    if path.is_file():
        if path.suffix != ".safetensors":
            raise AdapterParseError(
                f"Unsafe or unsupported weights format '{path.name}'. "
                "Only pickle-free .safetensors files are permitted."
            )
        cfg = path.parent / "adapter_config.json"
        fmt = AdapterFormat.PEFT_SAFETENSORS if cfg.is_file() else AdapterFormat.RAW_SAFETENSORS
        return path.parent, path, (cfg if cfg.is_file() else None), fmt

    if not path.is_dir():
        raise AdapterParseError(f"Adapter path does not exist: {path}")

    cfg = path / "adapter_config.json"
    default_weights = path / "adapter_model.safetensors"
    if default_weights.is_file():
        fmt = AdapterFormat.PEFT_SAFETENSORS if cfg.is_file() else AdapterFormat.RAW_SAFETENSORS
        return path, default_weights, (cfg if cfg.is_file() else None), fmt

    candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix == ".safetensors")
    if len(candidates) == 1:
        fmt = AdapterFormat.PEFT_SAFETENSORS if cfg.is_file() else AdapterFormat.RAW_SAFETENSORS
        return path, candidates[0], (cfg if cfg.is_file() else None), fmt

    if (path / "adapter_model.bin").is_file():
        raise AdapterParseError(
            f"Refusing unsafe pickle checkpoint 'adapter_model.bin' in {path}. "
            "Convert adapter weights to .safetensors first."
        )

    raise AdapterParseError(f"No .safetensors adapter weights found in directory: {path}")


def load_peft_adapter(
    path: Path,
    *,
    adapter_id: str | None = None,
    adapter_name: str | None = None,
    backend: MathBackend = "auto",
) -> LoadedLoRAWeights:
    """Safely load a LoRA adapter directory or `.safetensors` file into canonical schemas."""
    backend = validate_math_backend(backend)
    root_path, weights_path, config_path, adapter_format = _resolve_adapter_paths(path)
    st = parse_safetensors_header(weights_path)
    resolved_id = adapter_id or (root_path.name if path.is_dir() else weights_path.stem)

    cfg_data: dict[str, Any] = {}
    if config_path is not None:
        try:
            loaded_cfg: Any = json.loads(config_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterParseError(f"Malformed JSON in {config_path}: {exc}") from exc
        if not isinstance(loaded_cfg, dict):
            raise AdapterParseError(f"adapter_config.json must be a JSON object: {config_path}")
        cfg_data = cast(dict[str, Any], loaded_cfg)

    by_adapter: dict[str, dict[str, dict[str, SafetensorsTensorMeta]]] = {}
    dora_by_adapter: dict[str, list[str]] = {}
    for tname, tmeta in st.tensors.items():
        dora_match = _DORA_MAG_KEY_RE.match(tname)
        if dora_match is not None:
            dora_tag = dora_match.group("adapter") or "default"
            dora_by_adapter.setdefault(dora_tag, []).append(tname)
            continue
        match = _LORA_KEY_RE.match(tname)
        if match is None:
            continue
        prefix = match.group("prefix").rstrip(".")
        side = match.group("side")
        adapter_tag = match.group("adapter") or "default"
        layer_id = prefix or "__root__"
        adapter_bucket = by_adapter.setdefault(adapter_tag, {})
        layer_bucket = adapter_bucket.setdefault(layer_id, {})
        layer_bucket[side] = tmeta

    if not by_adapter:
        raise AdapterParseError(f"No paired lora_A / lora_B tensors found inside {weights_path}.")

    selected_adapter_tag: str
    if adapter_name is not None:
        if adapter_name not in by_adapter:
            raise AdapterParseError(
                f"Requested adapter_name='{adapter_name}' not found in {weights_path}; "
                f"available: {sorted(by_adapter)}."
            )
        selected_adapter_tag = adapter_name
    elif len(by_adapter) == 1:
        selected_adapter_tag = next(iter(by_adapter))
    elif "default" in by_adapter:
        selected_adapter_tag = "default"
    else:
        raise AdapterParseError(
            f"Multiple adapter namespaces {sorted(by_adapter)} found in {weights_path}; "
            "specify adapter_name explicitly."
        )

    dora_mag_keys = sorted(dora_by_adapter.get(selected_adapter_tag, []))
    if len(by_adapter) == 1 and not dora_mag_keys and "default" in dora_by_adapter:
        dora_mag_keys = sorted(dora_by_adapter["default"])

    grouped = by_adapter[selected_adapter_tag]

    raw_layers: list[
        tuple[str, SafetensorsTensorMeta, SafetensorsTensorMeta, int, int, int, int]
    ] = []
    layer_factors: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
    total_params = 0
    inferred_ranks: list[int] = []

    for layer_name in sorted(grouped):
        sides = grouped[layer_name]
        if "lora_A" not in sides or "lora_B" not in sides:
            raise AdapterParseError(
                f"Unpaired LoRA tensor for layer '{layer_name}' in {weights_path}: "
                f"found {sorted(sides)}."
            )
        meta_a = sides["lora_A"]
        meta_b = sides["lora_B"]
        if len(meta_a.shape) != 2 or len(meta_b.shape) != 2:
            raise AdapterParseError(
                f"LoRA factors for '{layer_name}' must be 2D; got {meta_a.shape}, {meta_b.shape}."
            )
        rank_a, in_features = meta_a.shape
        out_features, rank_b = meta_b.shape
        if rank_a != rank_b or rank_a <= 0:
            raise AdapterParseError(
                f"Rank mismatch in '{layer_name}': lora_A={meta_a.shape}, lora_B={meta_b.shape}."
            )
        if in_features <= 0 or out_features <= 0:
            raise AdapterParseError(
                f"Invalid empty LoRA feature dimension in '{layer_name}': "
                f"lora_A={meta_a.shape}, lora_B={meta_b.shape}."
            )

        mat_a = decode_tensor_2d(weights_path, meta_a, st.payload_offset, backend=backend)
        mat_b = decode_tensor_2d(weights_path, meta_b, st.payload_offset, backend=backend)
        param_count = rank_a * in_features + out_features * rank_b
        total_params += param_count
        inferred_ranks.append(rank_a)
        raw_layers.append(
            (layer_name, meta_a, meta_b, rank_a, in_features, out_features, param_count)
        )
        layer_factors[layer_name] = (mat_a, mat_b)

    cfg_rank = cfg_data.get("r")
    if "r" in cfg_data and cfg_rank is not None:
        if not _is_int_strict(cfg_rank) or cfg_rank <= 0:
            raise AdapterParseError(
                f"Invalid 'r' in {config_path}: expected positive integer, got {cfg_rank!r}."
            )
        rank = int(cfg_rank)
    else:
        rank = inferred_ranks[0]

    cfg_alpha = cfg_data.get("lora_alpha")
    config_alpha_missing = config_path is not None and (
        "lora_alpha" not in cfg_data or cfg_alpha is None
    )
    if "lora_alpha" in cfg_data and cfg_alpha is not None:
        if not _is_finite_number(cfg_alpha):
            raise AdapterParseError(
                f"Invalid 'lora_alpha' in {config_path}: expected finite number, got {cfg_alpha!r}."
            )
        lora_alpha = float(cfg_alpha)
    else:
        lora_alpha = float(rank)

    raw_dropout = cfg_data.get("lora_dropout", 0.0)
    if raw_dropout is None:
        lora_dropout = 0.0
    elif _is_finite_number(raw_dropout) and float(raw_dropout) >= 0.0:
        lora_dropout = float(raw_dropout)
    else:
        raise AdapterParseError(
            f"Invalid 'lora_dropout' in {config_path}: expected non-negative number, "
            f"got {raw_dropout!r}."
        )

    raw_rslora = cfg_data.get("use_rslora", False)
    if raw_rslora is not None and not isinstance(raw_rslora, bool):
        raise AdapterParseError(
            f"Invalid 'use_rslora' in {config_path}: expected boolean, got {raw_rslora!r}."
        )
    use_rslora = bool(raw_rslora) if raw_rslora is not None else False

    raw_dora = cfg_data.get("use_dora", False)
    if raw_dora is not None and not isinstance(raw_dora, bool):
        raise AdapterParseError(
            f"Invalid 'use_dora' in {config_path}: expected boolean, got {raw_dora!r}."
        )
    use_dora_cfg = bool(raw_dora) if raw_dora is not None else False
    use_dora = use_dora_cfg or bool(dora_mag_keys)

    scaling = lora_alpha / math.sqrt(float(rank)) if use_rslora else lora_alpha / float(rank)

    raw_rank_pattern = cfg_data.get("rank_pattern")
    rank_pattern: dict[str, int] = {}
    if raw_rank_pattern is not None:
        if not isinstance(raw_rank_pattern, dict):
            raise AdapterParseError(
                f"Invalid 'rank_pattern' in {config_path}: expected JSON object."
            )
        raw_rp_map = cast(dict[Any, Any], raw_rank_pattern)
        for pat_key, pat_rank in raw_rp_map.items():
            if not isinstance(pat_key, str) or not _is_int_strict(pat_rank) or pat_rank <= 0:
                raise AdapterParseError(
                    f"Invalid entry in 'rank_pattern' ({config_path}): "
                    f"{pat_key!r}: {pat_rank!r}; expected str -> positive int."
                )
            rank_pattern[pat_key] = int(pat_rank)

    raw_alpha_pattern = cfg_data.get("alpha_pattern")
    alpha_pattern: dict[str, float] = {}
    if raw_alpha_pattern is not None:
        if not isinstance(raw_alpha_pattern, dict):
            raise AdapterParseError(
                f"Invalid 'alpha_pattern' in {config_path}: expected JSON object."
            )
        raw_ap_map = cast(dict[Any, Any], raw_alpha_pattern)
        for pat_key, pat_alpha in raw_ap_map.items():
            if not isinstance(pat_key, str) or not _is_finite_number(pat_alpha):
                raise AdapterParseError(
                    f"Invalid entry in 'alpha_pattern' ({config_path}): "
                    f"{pat_key!r}: {pat_alpha!r}; expected str -> finite number."
                )
            alpha_pattern[pat_key] = float(pat_alpha)

    layer_specs: list[LoRALayerSpec] = []
    rank_mismatch_details: list[str] = []

    for (
        layer_name,
        meta_a,
        meta_b,
        layer_rank,
        in_features,
        out_features,
        param_count,
    ) in raw_layers:
        has_pat_rank, pat_rank_val = match_peft_pattern(
            layer_name, rank_pattern, config_path=config_path
        )
        has_pat_alpha, pat_alpha_val = match_peft_pattern(
            layer_name, alpha_pattern, config_path=config_path
        )

        if config_path is not None:
            if has_pat_rank:
                expected_layer_rank = int(pat_rank_val)
                if layer_rank != expected_layer_rank:
                    rank_mismatch_details.append(
                        f"{layer_name} "
                        f"(weights r={layer_rank} != rank_pattern r={expected_layer_rank})"
                    )
            elif "r" in cfg_data and cfg_rank is not None and layer_rank != rank:
                rank_mismatch_details.append(
                    f"{layer_name} (weights r={layer_rank} != config r={rank})"
                )

        if has_pat_alpha:
            layer_alpha = float(pat_alpha_val)
        elif config_path is None or config_alpha_missing:
            layer_alpha = float(layer_rank)
        else:
            layer_alpha = lora_alpha

        layer_scaling = (
            layer_alpha / math.sqrt(float(layer_rank))
            if use_rslora
            else layer_alpha / float(layer_rank)
        )

        layer_specs.append(
            LoRALayerSpec(
                layer_name=layer_name,
                module_type=_infer_module_type(layer_name),
                in_features=in_features,
                out_features=out_features,
                rank=layer_rank,
                lora_alpha=layer_alpha,
                scaling=layer_scaling,
                dtype=meta_a.dtype,
                a_key=meta_a.name,
                b_key=meta_b.name,
                a_shape=(layer_rank, in_features),
                b_shape=(out_features, layer_rank),
                param_count=param_count,
            )
        )

    config_rank_mismatch = bool(rank_mismatch_details)

    raw_targets: Any = cfg_data.get("target_modules", [])
    if isinstance(raw_targets, list):
        target_list = cast(list[Any], raw_targets)
        target_modules = [str(x) for x in target_list]
    elif isinstance(raw_targets, str):
        target_modules = [raw_targets]
    else:
        target_modules = sorted({ls.module_type for ls in layer_specs})

    base_model = cfg_data.get("base_model_name_or_path")
    spec = LoRAAdapterSpec(
        adapter_id=resolved_id,
        adapter_format=adapter_format,
        root_path=root_path.resolve(),
        weights_path=weights_path.resolve(),
        config_path=config_path.resolve() if config_path is not None else None,
        base_model_name_or_path=str(base_model) if base_model is not None else None,
        peft_type=str(cfg_data.get("peft_type", "LORA")),
        rank=rank,
        lora_alpha=lora_alpha,
        scaling=scaling,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora,
        use_dora=use_dora,
        bias=str(cfg_data.get("bias", "none")),
        target_modules=target_modules,
        total_lora_params=total_params,
        weights_sha256=sha256_file(weights_path),
        layers=layer_specs,
        metadata={
            "safetensors_metadata": st.metadata,
            "config_missing": config_path is None,
            "config_alpha_missing": config_alpha_missing,
            "config_rank_mismatch": config_rank_mismatch,
            "rank_mismatch_details": rank_mismatch_details,
            "rank_pattern": rank_pattern,
            "alpha_pattern": alpha_pattern,
            "use_dora_config": use_dora_cfg,
            "dora_magnitude_tensor_count": len(dora_mag_keys),
            "dora_magnitude_keys": dora_mag_keys,
            "selected_adapter_namespace": selected_adapter_tag,
            "available_adapter_namespaces": sorted(by_adapter),
        },
    )
    return LoadedLoRAWeights(spec=spec, layer_factors=layer_factors)
