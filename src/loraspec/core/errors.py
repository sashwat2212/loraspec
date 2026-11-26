"""Deterministic CLI exit codes and domain exceptions for `loraspec`."""

from __future__ import annotations

EXIT_PASS: int = 0
EXIT_AUDIT_FAIL: int = 1
EXIT_CONFIG_ERROR: int = 2
EXIT_BUDGET_EXCEEDED: int = 3
EXIT_RUNTIME_ERROR: int = 4
EXIT_PARITY_ERROR: int = 5


class LoraspecError(Exception):
    """Base exception carrying a deterministic CLI exit code."""

    exit_code: int = EXIT_RUNTIME_ERROR

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class AuditFailedError(LoraspecError):
    """Raised when an audit fails configured severity or grading thresholds."""

    exit_code: int = EXIT_AUDIT_FAIL


class ConfigError(LoraspecError):
    """Raised on invalid CLI arguments or malformed loraspec.toml."""

    exit_code: int = EXIT_CONFIG_ERROR


class AdapterParseError(LoraspecError):
    """Raised on malformed, truncated, or unsafe .safetensors / PEFT adapter files."""

    exit_code: int = EXIT_CONFIG_ERROR


class BudgetExceededError(LoraspecError):
    """Raised when GpuBudgetGuard INR/hour ceiling or --smoke cap triggers."""

    exit_code: int = EXIT_BUDGET_EXCEEDED


class RuntimeExecutionError(LoraspecError):
    """Raised when an optional CUDA/Triton/vLLM backend is unavailable or fails."""

    exit_code: int = EXIT_RUNTIME_ERROR


class KernelParityError(LoraspecError):
    """Raised when Triton kernel outputs exceed tolerance vs PyTorch reference."""

    exit_code: int = EXIT_PARITY_ERROR
