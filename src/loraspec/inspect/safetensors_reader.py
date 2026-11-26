"""Bounds-safe, pickle-free `.safetensors` binary reader/writer and PEFT adapter loader."""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loraspec.core.errors import AdapterParseError
from loraspec.core.manifest import sha256_file
from loraspec.core.schemas import AdapterFormat, LoRAAdapterSpec, LoRALayerSpec

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
) -> list[list[float]]:
    """Decode a 2D tensor slice from a `.safetensors` file without `pickle`."""
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

    flat: list[float]
    if meta.dtype == "F32":
        flat = [float(v[0]) for v in struct.iter_unpack("<f", raw)]
    elif meta.dtype == "F16":
        flat = [float(v[0]) for v in struct.iter_unpack("<e", raw)]
    elif meta.dtype == "F64":
        flat = [float(v[0]) for v in struct.iter_unpack("<d", raw)]
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


def load_peft_adapter(path: Path, *, adapter_id: str | None = None) -> LoadedLoRAWeights:
    """Safely load a LoRA adapter directory or `.safetensors` file into canonical schemas."""
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

    grouped: dict[str, dict[str, SafetensorsTensorMeta]] = {}
    for tname, tmeta in st.tensors.items():
        match = _LORA_KEY_RE.match(tname)
        if match is None:
            continue
        prefix = match.group("prefix").rstrip(".")
        side = match.group("side")
        layer_id = prefix or tname
        bucket = grouped.setdefault(layer_id, {})
        bucket[side] = tmeta

    if not grouped:
        raise AdapterParseError(f"No paired lora_A / lora_B tensors found inside {weights_path}.")

    layer_specs: list[LoRALayerSpec] = []
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

        mat_a = decode_tensor_2d(weights_path, meta_a, st.payload_offset)
        mat_b = decode_tensor_2d(weights_path, meta_b, st.payload_offset)
        param_count = rank_a * in_features + out_features * rank_b
        total_params += param_count
        inferred_ranks.append(rank_a)

        layer_specs.append(
            LoRALayerSpec(
                layer_name=layer_name,
                module_type=_infer_module_type(layer_name),
                in_features=in_features,
                out_features=out_features,
                rank=rank_a,
                dtype=meta_a.dtype,
                a_key=meta_a.name,
                b_key=meta_b.name,
                a_shape=(rank_a, in_features),
                b_shape=(out_features, rank_b),
                param_count=param_count,
            )
        )
        layer_factors[layer_name] = (mat_a, mat_b)

    cfg_rank = cfg_data.get("r")
    rank = int(cfg_rank) if isinstance(cfg_rank, int) and cfg_rank > 0 else inferred_ranks[0]
    cfg_alpha = cfg_data.get("lora_alpha")
    lora_alpha = float(cfg_alpha) if isinstance(cfg_alpha, (int, float)) else float(rank)
    use_rslora = bool(cfg_data.get("use_rslora", False))
    scaling = lora_alpha / math.sqrt(float(rank)) if use_rslora else lora_alpha / float(rank)

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
        lora_dropout=float(cfg_data.get("lora_dropout", 0.0)),
        use_rslora=use_rslora,
        use_dora=bool(cfg_data.get("use_dora", False)),
        bias=str(cfg_data.get("bias", "none")),
        target_modules=target_modules,
        total_lora_params=total_params,
        weights_sha256=sha256_file(weights_path),
        layers=layer_specs,
        metadata={
            "safetensors_metadata": st.metadata,
            "config_missing": config_path is None,
        },
    )
    return LoadedLoRAWeights(spec=spec, layer_factors=layer_factors)
