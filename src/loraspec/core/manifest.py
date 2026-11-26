"""Reproducibility manifest builder and host/accelerator probe utilities."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import platform
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loraspec import __version__
from loraspec.core.schemas import RunManifest


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a single file safely in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_git_sha(cwd: Path) -> str | None:
    """Return short HEAD git SHA if inside a git repository, else None."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd if cwd.is_dir() else cwd.parent,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    sha = proc.stdout.strip()
    return sha or None


def detect_accelerator_backend() -> str:
    """Detect best available compute backend without importing heavy CUDA libs unnecessarily."""
    has_numpy = importlib.util.find_spec("numpy") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    has_triton = importlib.util.find_spec("triton") is not None

    if has_torch:
        try:
            torch_mod: Any = importlib.import_module("torch")
            if bool(torch_mod.cuda.is_available()):
                return "triton_cuda" if has_triton else "torch_cuda"
            mps: Any = getattr(torch_mod.backends, "mps", None)
            if mps is not None and bool(mps.is_available()):
                return "torch_mps"
            return "torch_cpu"
        except Exception:
            pass

    if has_numpy:
        return "numpy_cpu"
    return "pure_python_cpu"


def build_run_manifest(
    *,
    adapter_sha256s: dict[str, str],
    root_hint: Path | None = None,
    rng_seed: int = 42,
    total_gpu_cost_inr: float = 0.0,
    cli_args: list[str] | None = None,
) -> RunManifest:
    """Construct immutable RunManifest stamped with exact host, git, and SHA-256 provenance."""
    cwd = root_hint if root_hint is not None else Path.cwd()
    return RunManifest(
        run_id=str(uuid.uuid4()),
        loraspec_version=__version__,
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
        git_sha=resolve_git_sha(cwd),
        host_os=platform.system().lower(),
        host_arch=platform.machine().lower(),
        accelerator_backend=detect_accelerator_backend(),
        adapter_sha256s=dict(sorted(adapter_sha256s.items())),
        rng_seed=rng_seed,
        total_gpu_cost_inr=round(total_gpu_cost_inr, 4),
        cli_args=list(cli_args) if cli_args is not None else [],
    )
